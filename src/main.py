from pathlib import Path

import streamlit as st
import torch
from tokenizers import Tokenizer

from tokenizer.jamo_bpe import JamoBPE
from tokenizer.hf_jamo_bpe import HFJamoBPE
from tokenizer.config import DEFAULT_VOCAB_SIZE, DEFAULT_MIN_FREQUENCY
from translation.model import Seq2SeqTransformer


BACKEND_DIRS = {
    ("small", "jamo"): "jamo",
    ("small", "full"): "full",
    ("large", "jamo"): "hf_jamo",
    ("large", "full"): "hf_full",
    ("large_16k", "jamo"): "hf_jamo_16k",
    ("large_16k", "full"): "hf_full_16k",
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
        ("small", "jamo"): JamoBPE(jamo_break=True),
        ("small", "full"): JamoBPE(jamo_break=False),
        ("large", "jamo"): HFJamoBPE(jamo_break=True),
        ("large", "full"): HFJamoBPE(jamo_break=False),
        ("large_16k", "jamo"): HFJamoBPE(jamo_break=True),
        ("large_16k", "full"): HFJamoBPE(jamo_break=False),
    }
    ensure_tokenizer(tokenizers[("small", "jamo")], output_dir / "jamo", input_dir)
    ensure_tokenizer(tokenizers[("small", "full")], output_dir / "full", input_dir)
    ensure_hf_tokenizer(tokenizers[("large", "jamo")], output_dir / "hf_jamo", input_dir)
    ensure_hf_tokenizer(tokenizers[("large", "full")], output_dir / "hf_full", input_dir)
    ensure_hf_tokenizer(tokenizers[("large_16k", "jamo")], output_dir / "hf_jamo_16k", input_dir)
    ensure_hf_tokenizer(tokenizers[("large_16k", "full")], output_dir / "hf_full_16k", input_dir)
    return tokenizers, input_dir, output_dir


TRANSLATOR_CKPTS = {
    "hf_jamo": "jamo_based.pt",
    "hf_full": "character_based.pt",
}


@st.cache_resource
def load_translator(kind: str):
    project_dir = Path(__file__).resolve().parents[1]
    ckpt_path = project_dir / "models" / TRANSLATOR_CKPTS[kind]
    if not ckpt_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    kr_tok = HFJamoBPE(jamo_break=(kind == "hf_jamo"))
    kr_tok.load(project_dir / "dicts" / kind)
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
    return model, kr_tok, en_tok, cfg, device


def run_translation(kind: str, text: str, max_len: int = 128) -> str | None:
    loaded = load_translator(kind)
    if loaded is None:
        return None
    model, kr_tok, en_tok, cfg, device = loaded

    src_ids = kr_tok.encode_ids(text)[:max_len] or [cfg["src_pad_id"]]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pm = src == cfg["src_pad_id"]
    ys = model.greedy_decode(src, src_pm, cfg["bos_id"], cfg["eos_id"], max_len=max_len)
    ids = ys[0].tolist()[1:]
    if cfg["eos_id"] in ids:
        ids = ids[: ids.index(cfg["eos_id"])]
    return en_tok.decode(ids)


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
        dataset = st.radio(
            "Dataset",
            ["small", "large", "large_16k"],
            horizontal=True,
            format_func=lambda v: {"small": "Small", "large": "Large", "large_16k": "Large (16k)"}[v],
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
                tokens = tokenizers[(dataset, mode)].encode(text.strip())
                label = f"{dataset.capitalize()} / {mode.capitalize()}"
                st.subheader(f"Tokens — {label} ({len(tokens)})")
                st.code(str(tokens), language="python")
            except Exception as e:
                st.error(f"Tokenization failed: {e}")

            if dataset == "large":
                kind = "hf_jamo" if mode == "jamo" else "hf_full"
                st.subheader(f"Translation — {mode.capitalize()}")
                try:
                    hyp = run_translation(kind, text.strip())
                    if hyp is None:
                        st.write(f"No checkpoint at models/{TRANSLATOR_CKPTS[kind]}")
                    else:
                        st.write(hyp)
                except Exception as e:
                    st.write(f"Translation failed: {e}")


main()
