# train_ulm.py
import time, os, torch, numpy as np
from transformers import TrainingArguments, Trainer
from dataset import UnitDataset, collate_fn
from model   import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("📚  Loading datasets")
train_ds = UnitDataset("units_train.csv")
dev_ds   = UnitDataset("units_dev.csv")

print("🤖  Building GPT-2 scratch model")
model = build_model().to(DEVICE)

args = TrainingArguments(
    output_dir="unit-lm-scratch",
    overwrite_output_dir=True,
    bf16=True,                         # Ampere 以上なら bf16
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,     # effective 64
    learning_rate=3e-4,
    warmup_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
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

# ---------- training ----------
print("🔎  Eval before")
print("   PPL :", np.exp(trainer.evaluate()['eval_loss']))
t0 = time.time()
trainer.train()
print("⏱  Total time : %.1f min" % ((time.time()-t0)/60))
print("   PPL after :", np.exp(trainer.evaluate()['eval_loss']))

trainer.save_model("unit-lm-scratch-final")
print("✅  Model saved to unit-lm-scratch-final/")