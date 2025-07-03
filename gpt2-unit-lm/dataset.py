# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_ID  = 0   # パディングに専用
SHIFT   = 1   # もとの 0-127 を 1-128 にずらす
MAX_LEN = 1024

class UnitDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = MAX_LEN):
        df = pd.read_csv(csv_path)
        self.seqs = []
        for row in df["text"]:
            ids = [int(x) + SHIFT for x in row.split()]
            if len(ids) >= 2:
                self.seqs.append(ids[: max_len + 1])   # +1 は label 用
        print(f"{csv_path}: {len(self.seqs):,} sequences")

    def __len__(self):  return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
            "labels"   : torch.tensor(seq[1:] , dtype=torch.long),
        }

def collate_fn(batch, pad_id: int = PAD_ID):
    L = max(len(x["input_ids"]) for x in batch)
    ids, lbl, msk = [], [], []
    for x in batch:
        l = len(x["input_ids"])
        pad_i = torch.full((L,), pad_id , dtype=torch.long)
        pad_l = torch.full((L,), -100  , dtype=torch.long)
        pad_m = torch.zeros(L,           dtype=torch.long)
        pad_i[:l] = x["input_ids"]; pad_l[:l] = x["labels"]; pad_m[:l] = 1
        ids.append(pad_i); lbl.append(pad_l); msk.append(pad_m)
    return {
        "input_ids"     : torch.stack(ids),
        "labels"        : torch.stack(lbl),
        "attention_mask": torch.stack(msk),
    }