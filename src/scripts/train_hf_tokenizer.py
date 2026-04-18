from itertools import chain
from pathlib import Path

from tqdm import tqdm

from ..tokenizer.hf_jamo_bpe import HFJamoBPE
from ..tokenizer.config import DEFAULT_MIN_FREQUENCY


# How to run: python -m src.scripts.train_hf_tokenizer


JAMO_BREAK = False

SUFFIX = "hf_full"
DATASET = "CocoRoF/cc-100-korean-processing"
USE_PARTS: list[int] | None = [0, 1]  # e.g. [0, 1, 2]; None = all
                  # JAMO_BREAK=True  -> indexes into cc100_hf_jamo.part*.txt (0..9)
                  # JAMO_BREAK=False -> indexes into HF dataset chunks (0..22)

VOCAB_SIZE = 16384
MIN_FREQUENCY = DEFAULT_MIN_FREQUENCY
OUTPUT_NAME = "hf_full_16k"


def file_line_iterator(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {path.name}"):
                line = line.rstrip("\n")
                if not line:
                    continue
                yield line


def hf_dataset_iterator(dataset):
    for example in tqdm(dataset, desc="Streaming HF dataset"):
        text = example.get("text", "")
        if not text.strip():
            continue
        yield text


def main():
    project_dir = Path(__file__).parents[2]
    output_dir = project_dir / "dicts" / OUTPUT_NAME

    if JAMO_BREAK:
        corpus_dir = project_dir / "datas" / "preprocessed"
        parts = sorted(corpus_dir.glob(f"cc100_{SUFFIX}.part*.txt"))
        if not parts:
            raise FileNotFoundError(f"No preprocessed parts found in {corpus_dir} matching cc100_{SUFFIX}.part*.txt")
        if USE_PARTS is not None:
            parts = [p for p in parts if int(p.stem.split(".part")[-1]) in set(USE_PARTS)]
            if not parts:
                raise FileNotFoundError(f"USE_PARTS={USE_PARTS} matched no files")
        print(f"Streaming from {len(parts)} preprocessed file(s):")
        for p in parts:
            print(f"  - {p.name}")
        iterator = file_line_iterator(parts)
    else:
        from datasets import load_dataset
        cache_dir = project_dir / "datas" / "huggingface"
        chunks = [f"chunk_{i:02d}" for i in range(23)]
        if USE_PARTS is not None:
            chunks = [f"chunk_{i:02d}" for i in USE_PARTS]
        print(f"Streaming HF dataset {DATASET} chunks={chunks}")
        datasets = [load_dataset(DATASET, c, split="train", cache_dir=str(cache_dir)) for c in chunks]
        iterator = hf_dataset_iterator(chain.from_iterable(datasets))

    print(f"Training tokenizer (jamo_break={JAMO_BREAK}, vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQUENCY})")
    tokenizer = HFJamoBPE(jamo_break=JAMO_BREAK)
    tokenizer.train_from_iterator(
        iterator,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
    )

    tokenizer.save(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    test_text = "안녕하세요 여러분"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens, to_hangul=True)
    print(f"\nTest: '{test_text}'")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Decoded: '{decoded}'")


if __name__ == "__main__":
    main()
