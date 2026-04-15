import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(_reconfigure):
    try:
        _reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from src.tokenizer.hf_jamo_bpe import HFJamoBPE
from src.translation.dataset import (
    KoEnPairDataset,
    build_pairs,
    make_collate_fn,
)
from src.translation.model import Seq2SeqTransformer


BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
D_MODEL = 512
N_HEAD = 8
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
DIM_FF = 1024
DROPOUT = 0.1
MAX_LEN = 128
TRAIN_LIMIT: int | None = None
VALID_LIMIT: int | None = None
WARMUP_STEPS = 1_000
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0
GRAD_CLIP = 1.0
SAMPLE_PRINT_EVERY_EPOCH = 5
LOG_STEP_EVERY = 100
LOG_FLUSH_EVERY = 1000

DICTS_ROOT = Path("dicts")
OUTPUT_ROOT = Path("runs")


def load_kr_tokenizer(kind: str) -> tuple[HFJamoBPE, dict]:
    assert kind in {"hf_jamo", "hf_full"}
    tok = HFJamoBPE(jamo_break=(kind == "hf_jamo"))
    tok.load(DICTS_ROOT / kind)
    ids = tok.ensure_special_tokens(tokens=("<pad>", "<unk>"))
    return tok, ids


def load_en_tokenizer() -> tuple[Tokenizer, dict]:
    en_tok = Tokenizer.from_pretrained("gpt2")
    en_tok.add_special_tokens(["<pad>", "<bos>", "<eos>"])
    return en_tok, {
        "<pad>": en_tok.token_to_id("<pad>"),
        "<bos>": en_tok.token_to_id("<bos>"),
        "<eos>": en_tok.token_to_id("<eos>"),
    }


def inverse_sqrt_schedule(step: int, warmup: int) -> float:
    step = max(step, 1)
    if step < warmup:
        return step / warmup
    return (warmup**0.5) / (step**0.5)


def build_loaders(kr_tok, en_tok, src_pad_id, tgt_pad_id, bos_id, eos_id):
    print(f"[data] loading train pairs (limit={TRAIN_LIMIT}) ...")
    train_pairs = build_pairs("train", limit=TRAIN_LIMIT)
    print(f"[data] train pairs: {len(train_pairs)}")
    print(f"[data] loading valid pairs (limit={VALID_LIMIT}) ...")
    valid_pairs = build_pairs("valid", limit=VALID_LIMIT)
    print(f"[data] valid pairs: {len(valid_pairs)}")

    def make_ds(pairs):
        return KoEnPairDataset(
            pairs, kr_tok, en_tok, src_pad_id, tgt_pad_id, bos_id, eos_id, max_len=MAX_LEN
        )

    collate = make_collate_fn(src_pad_id, tgt_pad_id)
    train_loader = DataLoader(
        make_ds(train_pairs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    valid_loader = DataLoader(
        make_ds(valid_pairs),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    return train_loader, valid_loader, valid_pairs


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out, src_pm, tgt_pm in loader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_pm = src_pm.to(device)
            tgt_pm = tgt_pm.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(src, tgt_in, src_pm, tgt_pm)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            ntok = int((tgt_out != criterion.ignore_index).sum().item())
            total_loss += float(loss.item()) * ntok
            total_tokens += ntok
    return total_loss / max(total_tokens, 1)


def translate_samples(model, kr_tok, en_tok, pairs, src_pad_id, bos_id, eos_id, device, n=3):
    model.eval()
    out = []
    for ko, en_ref in pairs[:n]:
        src_ids = kr_tok.encode_ids(ko)[:MAX_LEN] or [src_pad_id]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_pm = src == src_pad_id
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            ys = model.greedy_decode(src, src_pm, bos_id, eos_id, max_len=MAX_LEN)
        ids = ys[0].tolist()[1:]
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        hyp = en_tok.decode(ids)
        out.append((ko, en_ref, hyp))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kr-tokenizer", choices=["hf_jamo", "hf_full"], required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    kr_tok, kr_ids = load_kr_tokenizer(args.kr_tokenizer)
    src_pad_id = kr_ids["<pad>"]
    src_vocab = kr_tok.vocab_size()
    print(f"[kr] kind={args.kr_tokenizer} vocab={src_vocab} pad_id={src_pad_id}")

    en_tok, en_ids = load_en_tokenizer()
    tgt_pad_id = en_ids["<pad>"]
    bos_id = en_ids["<bos>"]
    eos_id = en_ids["<eos>"]
    tgt_vocab = en_tok.get_vocab_size()
    print(f"[en] vocab={tgt_vocab} pad={tgt_pad_id} bos={bos_id} eos={eos_id}")

    train_loader, valid_loader, valid_pairs = build_loaders(
        kr_tok, en_tok, src_pad_id, tgt_pad_id, bos_id, eos_id
    )

    model = Seq2SeqTransformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=N_ENC_LAYERS,
        num_decoder_layers=N_DEC_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN + 4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: inverse_sqrt_schedule(step, WARMUP_STEPS)
    )

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = OUTPUT_ROOT / args.kr_tokenizer / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    ckpt_path = out_dir / "best.pt"
    log_f = log_path.open("w", encoding="utf-8")
    print(f"[run] id={run_id} dir={out_dir}")

    def log(event: dict, flush: bool = True):
        event["ts"] = time.time()
        log_f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if flush:
            log_f.flush()

    log(
        {
            "type": "run_start",
            "run_id": run_id,
            "kr_tokenizer": args.kr_tokenizer,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "d_model": D_MODEL,
                "nhead": N_HEAD,
                "num_encoder_layers": N_ENC_LAYERS,
                "num_decoder_layers": N_DEC_LAYERS,
                "dim_feedforward": DIM_FF,
                "dropout": DROPOUT,
                "max_len": MAX_LEN,
                "train_limit": TRAIN_LIMIT,
                "valid_limit": VALID_LIMIT,
                "warmup_steps": WARMUP_STEPS,
                "label_smoothing": LABEL_SMOOTHING,
                "grad_clip": GRAD_CLIP,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
            },
        }
    )

    best_valid = math.inf
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        running = 0.0
        running_tok = 0
        for src, tgt_in, tgt_out, src_pm, tgt_pm in pbar:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_pm = src_pm.to(device)
            tgt_pm = tgt_pm.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(src, tgt_in, src_pm, tgt_pm)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            global_step += 1

            ntok = int((tgt_out != tgt_pad_id).sum().item())
            step_loss = float(loss.item())
            running += step_loss * ntok
            running_tok += ntok

            if LOG_STEP_EVERY and global_step % LOG_STEP_EVERY == 0:
                log(
                    {
                        "type": "step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": step_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "ntok": ntok,
                    },
                    flush=(global_step % LOG_FLUSH_EVERY == 0),
                )

            if global_step % 50 == 0:
                pbar.set_postfix(loss=f"{running/max(running_tok,1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running / max(running_tok, 1)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        log({"type": "epoch", "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "step": global_step})

        samples = translate_samples(
            model, kr_tok, en_tok, valid_pairs, src_pad_id, bos_id, eos_id, device, n=SAMPLE_PRINT_EVERY_EPOCH
        )
        for i, (ko, ref, hyp) in enumerate(samples):
            print(f"  [sample {i}] KO : {ko}")
            print(f"  [sample {i}] REF: {ref}")
            print(f"  [sample {i}] HYP: {hyp}")
        log({"type": "samples", "epoch": epoch, "samples": [{"ko": k, "ref": r, "hyp": h} for k, r, h in samples]})

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                    "kr_tokenizer": args.kr_tokenizer,
                    "config": {
                        "d_model": D_MODEL,
                        "nhead": N_HEAD,
                        "num_encoder_layers": N_ENC_LAYERS,
                        "num_decoder_layers": N_DEC_LAYERS,
                        "dim_feedforward": DIM_FF,
                        "dropout": DROPOUT,
                        "max_len": MAX_LEN,
                        "src_vocab": src_vocab,
                        "tgt_vocab": tgt_vocab,
                        "src_pad_id": src_pad_id,
                        "tgt_pad_id": tgt_pad_id,
                        "bos_id": bos_id,
                        "eos_id": eos_id,
                    },
                },
                ckpt_path,
            )
            print(f"[epoch {epoch}] saved best -> {ckpt_path}")

    log_f.close()


if __name__ == "__main__":
    main()
