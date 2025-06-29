import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

class UnitDataset(Dataset):
    def __init__(self, csv_file, max_length=511):
        self.df = pd.read_csv(csv_file)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        units_str = self.df.iloc[idx]['text']
        sequence = [int(x) for x in units_str.split()]

        sequence = sequence[:self.max_length+1]

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


def create_dataloaders(train_csv, dev_csv, batch_size=8, max_length=511):
    
    train_dataset = UnitDataset(train_csv, max_length=max_length)
    dev_dataset = UnitDataset(dev_csv, max_length=max_length)

    collate_fn = create_collate_fn(pad_token_id=0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2, #並列処理
        pin_memory=False,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2, #並列処理
        pin_memory=False,
    )

    return train_loader, dev_loader


def create_unit_language_model():
    config = GPT2Config(
        vocab_size=128,
        n_positions=512,
        n_embd=512, # 埋め込み次元
        n_layer=6, # Transformer層数
        n_head=8, # attentionhead数
    )

    model = GPT2LMHeadModel(config)
    return model, config

def train_unit_language_model():
    train_dataset = UnitDataset('units_train.csv', max_length=511)
    dev_dataset = UnitDataset('units_dev.csv', max_length=511)

    model, config = create_unit_language_model()

    data_collator = create_collate_fn(pad_token_id=0) # 0埋め

    training_args = TrainingArguments(
        output_dir='./unit-lm-results', 
        overwrite_output_dir = True, # 既存directoryを上書き

        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-4, # 学習率
        weight_decay=0.01,

        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,

        logging_dir='./unit-lm-logs',
        logging_steps=100,

        dataloader_num_workers=0, # 並列処理なし
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )


    eval_results = trainer.evaluate()
    print(f"学習前の損失: {eval_results['eval_loss']:.4f}")
    print(f"学習前のPerplexity: {np.exp(eval_results['eval_loss']):.2f}")
    
    trainer.train()
    final_eval = trainer.evaluate()
    print(f"最終損失: {final_eval['eval_loss']:.4f}")
    print(f"最終Perplexity: {np.exp(final_eval['eval_loss']):.2f}")
    
    trainer.save_model("./unit-lm-final")
    return trainer, model
