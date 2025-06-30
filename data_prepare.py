import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import time

class UnitDataset(Dataset):
    def __init__(self, csv_file, max_length=1024):
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


def create_dataloaders(train_csv, dev_csv, batch_size=8, max_length=1024):
    
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
        n_positions=1024, # 最大系列長（論文は3072)
        n_embd=1024, # 埋め込み次元
        n_layer=12, # Transformer層数
        n_head=16, # attentionhead数
        n_inner=4096,

        # 過学習防止
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model = GPT2LMHeadModel(config)
    return model, config


def create_qwen_unit_lm(vocab_size=128, model_name="Qwen/Qwen2.5-3B"):
    config = AutoConfig.from_pretrained(model_name)

    config.vocab_size = vocab_size
    config.max_position_embeddings = 1024

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(vocab_size)

    return model, config

def setup_training(model, train_dataset, dev_dataset):
    training_args = TrainingArguments(
        output_dif="./qwen-unit-lm",
        overwrite_outputdir=True,

        num_train_epochs=10,
        per_device_train_batch_size=4,  # Qwenは大きいのでバッチサイズ小さめ
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,   # 実効バッチサイズ = 4 * 8 = 32
        
        # 最適化
        learning_rate=5e-5,  # Qwenは小さめの学習率
        weight_decay=0.01,
        warmup_steps=1000,
        
        # 評価・保存
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps", 
        save_steps=1000,
        save_total_limit=3,
        
        # ログ
        logging_steps=100,
        report_to=None,  # wandbなどを使わない場合
        
        # メモリ効率化
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_datset=dev_dataset,
    )

    return trainer



def train_unit_language_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    train_dataset = UnitDataset('units_train.csv', max_length=1024)
    dev_dataset = UnitDataset('units_dev.csv', max_length=1024)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")


    model, config = create_unit_language_model()
    model.to(device)

    data_collator = create_collate_fn(pad_token_id=0) # 0埋め

    training_args = TrainingArguments(
        output_dir='./unit-lm-results', 
        overwrite_output_dir = True, # 既存directoryを上書き

        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-4, # 学習率
        weight_decay=0.01,

        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,

        logging_dir='./unit-lm-logs',
        logging_steps=100,

        dataloader_num_workers=4, # 並列処理
        remove_unused_columns=False,
        fp16=True,  # GPU高速化
        dataloader_pin_memory=True,  # GPU最適化
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    start_time = time.time()

    eval_results = trainer.evaluate()
    print(f"学習前の損失: {eval_results['eval_loss']:.4f}")
    print(f"学習前のPerplexity: {np.exp(eval_results['eval_loss']):.2f}")
    
    trainer.train()
    print("\\学習完了")
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n学習完了 (所要時間: {training_time/60:.1f}分)")
        
    # 最終評価
    print("\\n最終評価...")
    final_eval = trainer.evaluate()
    print(f"最終損失: {final_eval['eval_loss']:.4f}")
    print(f"最終Perplexity: {np.exp(final_eval['eval_loss']):.2f}")
    
    # モデル保存
    trainer.save_model("./unit-lm-final")
    
    return trainer, model

def quick_test():
    max_length = 1024
    
    # データセットテスト
    dataset = UnitDataset('units_train.csv', max_length=max_length)
    
    # サンプルデータ確認
    sample = dataset[0]
    
    # モデル作成テスト
    model, config = create_unit_language_model()
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_positions: {config.n_positions}")
    
    # collate_fn テスト
    collate_fn = create_collate_fn(pad_token_id=0)
    batch = [dataset[i] for i in range(3)]
    collated = collate_fn(batch)
    print(f"  バッチサイズ: {collated['input_ids'].shape}")
    

if __name__ == "__main__":
    print("🚀 Unit-based Language Model")

    model, config = create_qwen_unit_lm(
        vocab_size=128,
        model_name="Qwen/Qwen2.5-3B"
    )

    train_dataset = UnitDataset('units_train.csv')
    dev_dataset = UnitDataset('units_dev.csv')

    trainer = setup_training(model, train_dataset, dev_dataset)

    trainer.train()

    trainer.save_model("./qwen-unit-lm-final")

    
