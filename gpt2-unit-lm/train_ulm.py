import os, time, torch, numpy as np
from transformers import TrainingArguments, Trainer
from dataset import UnitDataset, collate_fn
from model_gpt2 import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📚  Loading datasets")
train_ds = UnitDataset("units_train.csv")
dev_ds   = UnitDataset("units_dev.csv")

print("🤖  Building GPT-2 scratch model")
model = build_model().to(DEVICE)

args = TrainingArguments(
    output_dir="ulm-gpt2-scratch",
    overwrite_output_dir=True,

    # －－－ Qwen 実験と同じハイパラ －－－
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,      # effective 64
    learning_rate=2e-4,                 # Qwen と合わせる
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

print("🔎  Eval before")
print("   PPL :", np.exp(trainer.evaluate()['eval_loss']))

t0 = time.time()
trainer.train()
print("⏱  Total time : %.1f min" % ((time.time()-t0)/60))

print("   PPL after :", np.exp(trainer.evaluate()['eval_loss']))
trainer.save_model("ulm-gpt2-scratch-final")
print("✅  Saved to ulm-gpt2-scratch-final/")