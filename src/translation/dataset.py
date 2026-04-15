import json
from pathlib import Path
from typing import Literal

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

DATA_ROOT = Path("datas/translation")


def translation_json_paths(split: Literal["train", "valid"]) -> list[Path]:
    if split == "train":
        sub = "1.Training/labeled/ko_en_train_set.json"
    else:
        sub = "2.Validation/labeled/ko_en_valid_set.json"
    return [DATA_ROOT / "tech" / "data" / sub, DATA_ROOT / "life" / "data" / sub]


def _iter_pairs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["data"] if isinstance(data, dict) else data
    for rec in records:
        ko = rec.get("ko")
        en = rec.get("en")
        if ko and en:
            yield ko, en


class KoEnPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        kr_tok,
        en_tok,
        src_pad_id: int,
        tgt_pad_id: int,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ):
        self.pairs = pairs
        self.kr_tok = kr_tok
        self.en_tok = en_tok
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ko, en = self.pairs[idx]
        src_ids = self.kr_tok.encode_ids(ko)[: self.max_len]
        if not src_ids:
            src_ids = [self.src_pad_id]
        tgt_body = self.en_tok.encode(en).ids[: self.max_len - 2]
        tgt_ids = [self.bos_id] + tgt_body + [self.eos_id]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def build_pairs(
    split: Literal["train", "valid"],
    limit: int | None = None,
    min_ko_len: int = 2,
    min_en_len: int = 2,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in translation_json_paths(split):
        if not path.exists():
            raise FileNotFoundError(f"missing translation json: {path}")
        for ko, en in _iter_pairs(path):
            if len(ko) < min_ko_len or len(en) < min_en_len:
                continue
            pairs.append((ko, en))
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs


def make_collate_fn(src_pad_id: int, tgt_pad_id: int):
    def collate(batch):
        srcs, tgts = zip(*batch)
        src = pad_sequence(list(srcs), batch_first=True, padding_value=src_pad_id)
        tgt = pad_sequence(list(tgts), batch_first=True, padding_value=tgt_pad_id)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_padding_mask = src == src_pad_id
        tgt_padding_mask = tgt_in == tgt_pad_id
        return src, tgt_in, tgt_out, src_padding_mask, tgt_padding_mask

    return collate
