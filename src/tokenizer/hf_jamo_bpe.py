from pathlib import Path
import json

from kss import Kss
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from .config import DEFAULT_VOCAB_SIZE, DEFAULT_MIN_FREQUENCY

EOW = "</w>"


class HFJamoBPE:
    def __init__(self, jamo_break=True):
        self.h2hcj = Kss("h2hcj")
        self.hcj2h = Kss("hcj2h")
        self.jamo_break = jamo_break
        self.vocab = {}
        self._tokenizer = None

    def read_txt_files(self, input_dir):
        texts = []
        for path in Path(input_dir).rglob("*.txt"):
            text = path.read_text(encoding="utf-8")
            texts.append(text)
        return texts

    def preprocess(self, texts):
        out = []
        for text in texts:
            if self.jamo_break:
                text = self.h2hcj(text)
            out.append(text)
        return out

    def _create_tokenizer_and_trainer(self, vocab_size, min_frequency):
        tokenizer = Tokenizer(models.BPE(end_of_word_suffix=EOW))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[],
            end_of_word_suffix=EOW,
            show_progress=True,
        )
        return tokenizer, trainer

    def train(self, texts, vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=DEFAULT_MIN_FREQUENCY):
        tokenizer, trainer = self._create_tokenizer_and_trainer(vocab_size, min_frequency)
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def train_from_iterator(self, iterator, vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=DEFAULT_MIN_FREQUENCY):
        tokenizer, trainer = self._create_tokenizer_and_trainer(vocab_size, min_frequency)
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def encode(self, text):
        if self.jamo_break:
            text = self.h2hcj(text)

        encoding = self._tokenizer.encode(text)
        return encoding.tokens

    def encode_ids(self, text):
        assert self._tokenizer is not None, "tokenizer not loaded"
        if self.jamo_break:
            text = self.h2hcj(text)
        return self._tokenizer.encode(text).ids

    def ensure_special_tokens(self, tokens=("<pad>", "<unk>", "<bos>", "<eos>")):
        assert self._tokenizer is not None, "tokenizer not loaded"
        self._tokenizer.add_special_tokens(list(tokens))
        self.vocab = self._tokenizer.get_vocab()
        return {t: self._tokenizer.token_to_id(t) for t in tokens}

    def vocab_size(self):
        assert self._tokenizer is not None, "tokenizer not loaded"
        return self._tokenizer.get_vocab_size()

    def decode(self, tokens, to_hangul=False):
        text = "".join(tokens).replace(EOW, " ").strip()
        if to_hangul and self.jamo_break:
            text = self.hcj2h(text)
        return text

    def save(self, output_dir, preprocessed_texts=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save(str(output_dir / "tokenizer.json"))

        (output_dir / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if preprocessed_texts:
            (output_dir / "corpus_h2hcj.txt").write_text(
                "\n".join(preprocessed_texts),
                encoding="utf-8",
            )

    def load(self, output_dir):
        output_dir = Path(output_dir)
        tokenizer_path = output_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in: {output_dir}")

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab = self._tokenizer.get_vocab()
