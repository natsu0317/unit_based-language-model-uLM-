# train_ulm.pyprint("=== THIS IS THE NEW VERSION ===")   # ←1行目に追加
import os
import time, torch, numpy as np
from transformers import TrainingArguments, Trainer
from dataset import UnitDataset, collate_fn
from model   import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📚  Loading datasets …")
train_ds = UnitDataset("units_train.csv")
dev_ds   = UnitDataset("units_dev.csv")

print("🤖  Building model …")
model = build_model().to(DEVICE)

args = TrainingArguments(
    output_dir="ulm-qwen3b",
    overwrite_output_dir=True,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,     # effective 64
    learning_rate=2e-4,
    warmup_steps=2000,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=lambda x: collate_fn(x, pad_id=0),
)

print("🔎  Evaluating before training …")
# m0 = trainer.evaluate()
# print(f"   PPL(before) : {np.exp(m0['eval_loss']):.2f}")

t0 = time.time()
checkpoint = "ulm-qwen3b/checkpoint-9000"

if checkpoint and os.path.isdir(checkpoint):
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()
print(f"⏱  Training time : {(time.time()-t0)/60:.1f} min")

m1 = trainer.evaluate()
print(f"   PPL(after)  : {np.exp(m1['eval_loss']):.2f}")

# LoRA アダプタのみ (再学習用)
trainer.save_model("ulm-qwen3b-lora")

# LoRA をベースに統合して保存 (推論用)
print("💾  Merging LoRA and saving …")
merged = trainer.model.merge_and_unload()
merged.save_pretrained("ulm-qwen3b-merged")