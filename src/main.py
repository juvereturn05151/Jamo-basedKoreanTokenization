from tokenizer.jamo_bpe import JamoBPE
from pathlib import Path
import tkinter as tk

from app import JamoBPEApp


def main():
    PROJECT_DIR = Path(__file__).parents[1]

    INPUT_DIR = PROJECT_DIR / "datas" / "train"
    OUTPUT_DIR = PROJECT_DIR / "dicts"

    # -----------------------------
    # Tokenizers
    # -----------------------------
    tokenizer = JamoBPE()
    full_tokenizer = JamoBPE(jamo_break=False)

    # -----------------------------
    # Training (your original logic)
    # -----------------------------
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

    tokenizer.save(OUTPUT_DIR / "jamo", preprocessed_texts)
    full_tokenizer.save(OUTPUT_DIR / "full", preprocessed_texts)

    # -----------------------------
    # GUI
    # -----------------------------
    root = tk.Tk()

    app = JamoBPEApp(
        root=root,
        tokenizer=tokenizer,
        full_tokenizer=full_tokenizer
    )

    root.mainloop()


if __name__ == "__main__":
    main()