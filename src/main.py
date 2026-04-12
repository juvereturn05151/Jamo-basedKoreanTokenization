from pathlib import Path

import streamlit as st

from tokenizer.jamo_bpe import JamoBPE
from tokenizer.hf_jamo_bpe import HFJamoBPE
from tokenizer.config import DEFAULT_VOCAB_SIZE, DEFAULT_MIN_FREQUENCY


BACKEND_DIRS = {
    ("local", "jamo"): "jamo",
    ("local", "full"): "full",
    ("hf", "jamo"): "hf_jamo",
    ("hf", "full"): "hf_full",
}


def ensure_tokenizer(tokenizer, output_dir, input_dir, vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=DEFAULT_MIN_FREQUENCY):
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


def ensure_hf_tokenizer(tokenizer, output_dir, input_dir, vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=DEFAULT_MIN_FREQUENCY):
    tokenizer_path = output_dir / "tokenizer.json"

    if tokenizer_path.exists():
        tokenizer.load(output_dir)
        print(f"Loaded HF tokenizer from {output_dir}")
    else:
        print(f"Training HF tokenizer for {output_dir}...")
        raw_texts = tokenizer.read_txt_files(input_dir)
        preprocessed_texts = tokenizer.preprocess(raw_texts)

        tokenizer.train(
            preprocessed_texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        tokenizer.save(output_dir, preprocessed_texts)
        print(f"Saved HF tokenizer to {output_dir}")


@st.cache_resource
def load_tokenizers():
    project_dir = Path(__file__).resolve().parents[1]
    input_dir = project_dir / "datas" / "train"
    output_dir = project_dir / "dicts"

    tokenizers = {
        ("local", "jamo"): JamoBPE(jamo_break=True),
        ("local", "full"): JamoBPE(jamo_break=False),
        ("hf", "jamo"): HFJamoBPE(jamo_break=True),
        ("hf", "full"): HFJamoBPE(jamo_break=False),
    }
    ensure_tokenizer(tokenizers[("local", "jamo")], output_dir / "jamo", input_dir)
    ensure_tokenizer(tokenizers[("local", "full")], output_dir / "full", input_dir)
    ensure_hf_tokenizer(tokenizers[("hf", "jamo")], output_dir / "hf_jamo", input_dir)
    ensure_hf_tokenizer(tokenizers[("hf", "full")], output_dir / "hf_full", input_dir)
    return tokenizers, input_dir, output_dir


def retrain_all(tokenizers, input_dir, output_dir):
    sample = next(iter(tokenizers.values()))
    raw_texts = sample.read_txt_files(input_dir)
    for key, tokenizer in tokenizers.items():
        texts = tokenizer.preprocess(raw_texts)
        tokenizer.train(
            texts,
            vocab_size=DEFAULT_VOCAB_SIZE,
            min_frequency=DEFAULT_MIN_FREQUENCY,
        )
        tokenizer.save(output_dir / BACKEND_DIRS[key], texts)


def main():
    st.set_page_config(page_title="Jamo BPE Tokenizer", layout="wide")
    st.title("Jamo BPE Tokenizer")

    tokenizers, input_dir, output_dir = load_tokenizers()

    _, retrain_col = st.columns([4, 1])
    with retrain_col:
        run_retrain = st.button(
            "Retrain Tokenizers",
            icon=":material/warning:",
        )

    text = st.text_area("Enter Korean text", value="안녕하세요 여러분", height=100)

    col1, col2 = st.columns(2)
    with col1:
        backend = st.radio(
            "Backend",
            ["local", "hf"],
            horizontal=True,
            format_func=lambda v: {"local": "Local", "hf": "HuggingFace"}[v],
        )
    with col2:
        mode = st.radio(
            "Mode",
            ["jamo", "full"],
            horizontal=True,
            format_func=str.capitalize,
        )

    run_exec = st.button("Execute", type="primary")

    if run_retrain:
        with st.spinner("Retraining tokenizers..."):
            try:
                retrain_all(tokenizers, input_dir, output_dir)
                st.success("Retraining complete")
            except Exception as e:
                st.error(f"Retraining failed: {e}")

    if run_exec:
        if not text.strip():
            st.warning("No input provided.")
        else:
            try:
                tokens = tokenizers[(backend, mode)].encode(text.strip())
                label = f"{backend.capitalize()} / {mode.capitalize()}"
                st.subheader(f"Tokens — {label} ({len(tokens)})")
                st.code(str(tokens), language="python")
            except Exception as e:
                st.error(f"Tokenization failed: {e}")


main()
