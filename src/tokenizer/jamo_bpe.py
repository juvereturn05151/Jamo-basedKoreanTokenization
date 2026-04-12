from pathlib import Path
from collections import Counter
import json

from kss import Kss

from .config import DEFAULT_VOCAB_SIZE, DEFAULT_MIN_FREQUENCY

EOW = "</w>"


class JamoBPE:
    def __init__(self, jamo_break=True):
        self.h2hcj = Kss("h2hcj")
        self.hcj2h = Kss("hcj2h")
        self.jamo_break = jamo_break
        self.merges = []
        self.vocab = {}

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

    def build_word_freqs(self, texts):
        word_freqs = Counter()

        for text in texts:
            for word in text.split():
                symbols = tuple(list(word) + [EOW])
                word_freqs[symbols] += 1

        return word_freqs

    def get_pair_stats(self, word_freqs):
        pair_stats = Counter()

        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_stats[pair] += freq

        return pair_stats

    def merge_pair_in_word(self, symbols, pair):
        new_symbols = []
        i = 0

        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1

        return tuple(new_symbols)

    def merge_corpus(self, word_freqs, pair):
        new_word_freqs = Counter()

        for symbols, freq in word_freqs.items():
            new_symbols = self.merge_pair_in_word(symbols, pair)
            new_word_freqs[new_symbols] += freq

        return new_word_freqs

    def collect_vocab(self, word_freqs):
        vocab = set()
        for symbols in word_freqs:
            vocab.update(symbols)
        return sorted(vocab)

    def train(self, texts, vocab_size=DEFAULT_VOCAB_SIZE, min_frequency=DEFAULT_MIN_FREQUENCY):
        word_freqs = self.build_word_freqs(texts)
        self.merges = []

        while True:
            vocab = self.collect_vocab(word_freqs)
            if len(vocab) >= vocab_size:
                break

            pair_stats = self.get_pair_stats(word_freqs)
            if not pair_stats:
                break

            best_pair, best_freq = pair_stats.most_common(1)[0]
            if best_freq < min_frequency:
                break

            self.merges.append(best_pair)
            word_freqs = self.merge_corpus(word_freqs, best_pair)

        final_vocab = self.collect_vocab(word_freqs)
        self.vocab = {token: i for i, token in enumerate(final_vocab)}

    def encode_word(self, word):
        symbols = list(word) + [EOW]

        while True:
            merged = False

            for pair in self.merges:
                i = 0
                new_symbols = []

                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                        new_symbols.append(symbols[i] + symbols[i + 1])
                        i += 2
                        merged = True
                    else:
                        new_symbols.append(symbols[i])
                        i += 1

                symbols = new_symbols

            if not merged:
                break

        return symbols

    def encode(self, text):
        if self.jamo_break:
            text = self.h2hcj(text)

        tokens = []
        for word in text.split():
            tokens.extend(self.encode_word(word))

        return tokens

    def decode(self, tokens, to_hangul=False):
        text = "".join(tokens).replace(EOW, " ").strip()
        if to_hangul:
            if self.jamo_break:
                text = self.hcj2h(text)
            return text

        return text

    def save(self, output_dir, preprocessed_texts):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "corpus_h2hcj.txt").write_text(
            "\n".join(preprocessed_texts),
            encoding="utf-8"
        )

        (output_dir / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Convert tuples to lists for JSON compatibility
        serializable_merges = [list(pair) for pair in self.merges]
        (output_dir / "merges.json").write_text(
            json.dumps(serializable_merges, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def load(self, output_dir):
        output_dir = Path(output_dir)

        vocab_path = output_dir / "vocab.json"
        merges_path = output_dir / "merges.json"

        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(f"Missing tokenizer files in: {output_dir}")

        self.vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

        loaded_merges = json.loads(merges_path.read_text(encoding="utf-8"))
        self.merges = [tuple(pair) for pair in loaded_merges]