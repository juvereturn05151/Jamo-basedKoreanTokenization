import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.tokenizer.hf_jamo_bpe import HFJamoBPE
from src.translation.model import Seq2SeqTransformer


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    kr_kind = ckpt["kr_tokenizer"]

    kr_tok = HFJamoBPE(jamo_break=(kr_kind == "hf_jamo"))
    kr_tok.load(Path("dicts") / kr_kind)
    kr_tok.ensure_special_tokens(tokens=("<pad>", "<unk>"))

    en_tok = Tokenizer.from_pretrained("gpt2")
    en_tok.add_special_tokens(["<pad>", "<bos>", "<eos>"])

    model = Seq2SeqTransformer(
        src_vocab=cfg["src_vocab"],
        tgt_vocab=cfg["tgt_vocab"],
        src_pad_id=cfg["src_pad_id"],
        tgt_pad_id=cfg["tgt_pad_id"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"] + 4,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, kr_tok, en_tok, cfg


def translate(model, kr_tok, en_tok, cfg, text: str, device: torch.device, max_len: int = 128) -> str:
    src_ids = kr_tok.encode_ids(text)[:max_len] or [cfg["src_pad_id"]]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pm = src == cfg["src_pad_id"]
    ys = model.greedy_decode(src, src_pm, cfg["bos_id"], cfg["eos_id"], max_len=max_len)
    ids = ys[0].tolist()[1:]
    if cfg["eos_id"] in ids:
        ids = ids[: ids.index(cfg["eos_id"])]
    return en_tok.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kr_tok, en_tok, cfg = load_checkpoint(args.checkpoint, device)
    hyp = translate(model, kr_tok, en_tok, cfg, args.text, device)
    print(hyp)


if __name__ == "__main__":
    main()
