from pathlib import Path
import tkinter as tk

from tokenizer.jamo_bpe import JamoBPE
from app import JamoBPEApp


def ensure_tokenizer(tokenizer, output_dir, input_dir, vocab_size=32768, min_frequency=2):
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"

    if vocab_path.exists() and merges_path.exists():
        tokenizer.load(output_dir)
        print(f"Loaded tokenizer from {output_dir}")
    else:
        print(f"Training tokenizer for {output_dir}...")
        raw_texts = tokenizer.read_txt_files(input_dir)
        preprocessed_texts = tokenizer.preprocess(raw_texts)

        tokenizer.train(
            preprocessed_texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        tokenizer.save(output_dir, preprocessed_texts)
        print(f"Saved tokenizer to {output_dir}")


def main():
    project_dir = Path(__file__).parents[1]
    input_dir = project_dir / "datas" / "train"
    output_dir = project_dir / "dicts"

    tokenizer = JamoBPE()
    full_tokenizer = JamoBPE(jamo_break=False)

    ensure_tokenizer(tokenizer, output_dir / "jamo", input_dir)
    ensure_tokenizer(full_tokenizer, output_dir / "full", input_dir)

    root = tk.Tk()
    app = JamoBPEApp(
        root=root,
        tokenizer=tokenizer,
        full_tokenizer=full_tokenizer,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    root.mainloop()


if __name__ == "__main__":
    main()