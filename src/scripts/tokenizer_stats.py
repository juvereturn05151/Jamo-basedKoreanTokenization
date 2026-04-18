import csv
import statistics
from pathlib import Path

from tqdm import tqdm

from ..tokenizer.jamo_bpe import JamoBPE
from ..tokenizer.hf_jamo_bpe import HFJamoBPE


# How to run: python -m src.scripts.tokenizer_stats


CHUNK_INDEX = 5
MAX_SAMPLES: int | None = 5000  # None = whole chunk
DATASET = "CocoRoF/cc-100-korean-processing"
TIKTOKEN_ENCODING = "o200k_base"  # GPT-4o encoding

BACKEND_DIRS = {
    ("local", "jamo"): "jamo",
    ("local", "full"): "full",
    ("hf", "jamo"): "hf_jamo",
    ("hf", "full"): "hf_full",
    ("hf", "jamo_16k"): "hf_jamo_16k",
    ("hf", "full_16k"): "hf_full_16k",
}


def load_tokenizers(dicts_root):
    tokenizers = {
        ("local", "jamo"): JamoBPE(jamo_break=True),
        ("local", "full"): JamoBPE(jamo_break=False),
        ("hf", "jamo"): HFJamoBPE(jamo_break=True),
        ("hf", "full"): HFJamoBPE(jamo_break=False),
        ("hf", "jamo_16k"): HFJamoBPE(jamo_break=True),
        ("hf", "full_16k"): HFJamoBPE(jamo_break=False),
    }
    for key, tok in tokenizers.items():
        tok.load(dicts_root / BACKEND_DIRS[key])

    import tiktoken
    tokenizers[("tiktoken", TIKTOKEN_ENCODING)] = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return tokenizers


def summarize(lengths):
    n = len(lengths)
    ordered = sorted(lengths)
    stdev = statistics.pstdev(lengths)
    return {
        "samples": n,
        "total": sum(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "stdev": stdev,
        "var": stdev * stdev,
        "min": ordered[0],
        "p95": ordered[min(n - 1, int(0.95 * n))],
        "max": ordered[-1],
    }


STAT_COLUMNS = ["samples", "total", "mean", "median", "stdev", "var", "min", "p95", "max"]


def save_csv(stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", *STAT_COLUMNS])
        for key, s in stats.items():
            writer.writerow([f"{key[0]}/{key[1]}", *(s[c] for c in STAT_COLUMNS)])


def print_table(stats):
    headers = ["tokenizer", "samples", "total", "mean", "median", "stdev", "var", "min", "p95", "max"]
    rows = []
    for key, s in stats.items():
        rows.append([
            f"{key[0]}/{key[1]}",
            f"{s['samples']}",
            f"{s['total']}",
            f"{s['mean']:.2f}",
            f"{s['median']:.1f}",
            f"{s['stdev']:.2f}",
            f"{s['var']:.2f}",
            f"{s['min']}",
            f"{s['p95']}",
            f"{s['max']}",
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main():
    project_dir = Path(__file__).parents[2]
    dicts_root = project_dir / "dicts"
    cache_dir = project_dir / "datas" / "huggingface"

    print(f"Loading tokenizers from {dicts_root}")
    tokenizers = load_tokenizers(dicts_root)

    from datasets import load_dataset
    chunk_name = f"chunk_{CHUNK_INDEX:02d}"
    print(f"Loading {DATASET} / {chunk_name}")
    ds = load_dataset(DATASET, chunk_name, split="train", cache_dir=str(cache_dir))

    lengths = {key: [] for key in tokenizers}
    seen = 0
    total = min(len(ds), MAX_SAMPLES) if MAX_SAMPLES is not None else len(ds)
    for example in tqdm(ds, total=total, desc=f"Tokenizing {chunk_name}"):
        text = example.get("text", "")
        if not text.strip():
            continue
        for key, tok in tokenizers.items():
            lengths[key].append(len(tok.encode(text)))
        seen += 1
        if MAX_SAMPLES is not None and seen >= MAX_SAMPLES:
            break

    stats = {key: summarize(lst) for key, lst in lengths.items()}

    print()
    print(f"Token-count statistics on {DATASET} / {chunk_name}")
    print()
    print_table(stats)

    csv_path = Path(__file__).parent / "stats" / f"tokenizer_stats_{chunk_name}.csv"
    save_csv(stats, csv_path)
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
