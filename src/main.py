from tokenizer.jamo_bpe import JamoBPE
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]

INPUT_DIR = PROJECT_DIR / "datas" / "train"
OUTPUT_DIR = PROJECT_DIR / "dicts"

tokenizer = JamoBPE()
full_tokenizer = JamoBPE(jamo_break=False)

raw_texts = tokenizer.read_txt_files(INPUT_DIR)
preprocessed_texts = tokenizer.preprocess(raw_texts)

tokenizer.train(
    preprocessed_texts,
    vocab_size=2000,
    min_frequency=2,
)
full_tokenizer.train(
    preprocessed_texts,
    vocab_size=2000,
    min_frequency=2,
)

tokenizer.save(OUTPUT_DIR/"jamo", preprocessed_texts)
full_tokenizer.save(OUTPUT_DIR/"full", preprocessed_texts)

tokens = tokenizer.encode("안녕하세요 여러분")
full_tokens = full_tokenizer.encode("안녕하세요 여러분")

print("Tokens:", tokens)
print("Full Tokens:", full_tokens)