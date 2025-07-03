# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_ID  = 0          # padding
SHIFT   = 1          # もとの unit(0-127) を 1-128 にシフト
MAX_LEN = 1024       # +1 で 1025 トークンまで保持

class UnitDataset(Dataset):
    def __init__(self, csv_path, max_len: int = MAX_LEN):
        df = pd.read_csv(csv_path)
        self.seqs = [] # 数字列
        for row in df["text"]:
            ids = [int(x) + SHIFT for x in row.split()]
            if len(ids) >= 2:                          # 1 トークン列は除外
                self.seqs.append(ids[: max_len + 1])   # +1 はラベル用
        print(f"{csv_path}: {len(self.seqs):,} sequences")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
            "labels"   : torch.tensor(seq[1:] , dtype=torch.long), # 正解(1つずつ右シフト)
        }


def collate_fn(batch, pad_id: int = PAD_ID):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids, lbl, msk = [], [], []
    for x in batch:
        L = len(x["input_ids"])
        pad_i = torch.full((max_len,), pad_id, dtype=torch.long)
        pad_l = torch.full((max_len,), -100,  dtype=torch.long) # label=-100は無視
        pad_m = torch.zeros(max_len,         dtype=torch.long) # 0埋めしたところは無視

        pad_i[:L] = x["input_ids"]
        pad_l[:L] = x["labels"]
        pad_m[:L] = 1
        ids.append(pad_i); lbl.append(pad_l); msk.append(pad_m)

    return {
        "input_ids"     : torch.stack(ids),
        "labels"        : torch.stack(lbl),
        "attention_mask": torch.stack(msk),
    }