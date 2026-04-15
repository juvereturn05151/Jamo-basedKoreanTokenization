import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=src_pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=tgt_pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab)

    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_emb(src) * math.sqrt(self.d_model)
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype("cuda"))
        return self.dropout(self.pos(x))

    def _embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype("cuda"))
        return self.dropout(self.pos(x))

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_e = self._embed_src(src)
        tgt_e = self._embed_tgt(tgt_in)
        tgt_len = tgt_in.size(1)
        causal = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=src.device), diagonal=1
        )
        out = self.transformer(
            src_e,
            tgt_e,
            src_mask=None,
            tgt_mask=causal,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        self.eval()
        device = src.device
        batch = src.size(0)
        memory = self.transformer.encoder(
            self._embed_src(src), src_key_padding_mask=src_padding_mask
        )
        ys = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tl = ys.size(1)
            causal = torch.triu(
                torch.ones(tl, tl, dtype=torch.bool, device=device), diagonal=1
            )
            out = self.transformer.decoder(
                self._embed_tgt(ys),
                memory,
                tgt_mask=causal,
                memory_key_padding_mask=src_padding_mask,
            )
            logits = self.generator(out[:, -1])
            next_tok = logits.argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break
        return ys
