import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UnitDataset(Dataset):
    def __init__(self, csv_file, max_length=512):
        self.df = pd.read_csv(csv_file)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        units_str = self.df.iloc[idx]['text']
        sequence = [int(x) for x in units_str.split()]
        input_ids = sequence[:-1]  # 最後を除く
        labels = sequence[1:]      # 最初を除く
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
def create_collate_fn(pad_token_id=0):
    
    def collate_fn(batch):
        # バッチ内の最大長を取得
        max_input_len = max(len(item['input_ids']) for item in batch)
        max_label_len = max(len(item['labels']) for item in batch)
        max_len = max(max_input_len, max_label_len)
        
        # パディング処理
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        
        for item in batch:
            input_ids = item['input_ids']
            labels = item['labels']
            
            # 入力パディング
            input_padded = torch.full((max_len,), pad_token_id, dtype=torch.long)
            input_padded[:len(input_ids)] = input_ids
            padded_inputs.append(input_padded)
            
            # ラベルパディング
            label_padded = torch.full((max_len,), -100, dtype=torch.long)
            label_padded[:len(labels)] = labels
            padded_labels.append(label_padded)
            
            # アテンションマスク
            mask = torch.zeros(max_len, dtype=torch.long)
            mask[:len(input_ids)] = 1
            attention_masks.append(mask)
        
        return {
            'input_ids': torch.stack(padded_inputs),
            'labels': torch.stack(padded_labels),
            'attention_mask': torch.stack(attention_masks)
        }
    
    return collate_fn


def create_dataloaders(train_csv, dev_csv, batch_size=8, max_length=512):
    
    train_dataset = UnitDataset(train_csv, max_length=max_length)
    dev_dataset = UnitDataset(dev_csv, max_length=max_length)

    collate_fn = create_collate_fn(pad_token_id=0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2, #並列処理
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2, #並列処理
        pin_memory=True,
    )

    return train_loader, dev_loader

