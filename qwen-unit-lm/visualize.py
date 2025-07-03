import json, pandas as pd, matplotlib.pyplot as plt, numpy as np

state = json.load(open("ulm-qwen3b/checkpoint-13000/trainer_state.json"))
logs  = pd.DataFrame(state["log_history"])

# ---- 訓練損失
plt.figure(figsize=(8,4))
train = logs.dropna(subset=["loss"])
plt.plot(train["step"], train["loss"], label="train_loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.savefig("train_loss.png")

# ---- 評価損失
plt.figure(figsize=(8,4))
eval_ = logs.dropna(subset=["eval_loss"])
plt.plot(eval_["step"], eval_["eval_loss"], label="eval_loss", color="orange")
plt.xlabel("step"); plt.ylabel("eval_loss"); plt.legend(); plt.tight_layout()
plt.savefig("eval_loss.png")

# ---- 学習率
plt.figure(figsize=(8,4))
plt.plot(train["step"], train["learning_rate"], label="lr", color="green")
plt.xlabel("step"); plt.ylabel("learning_rate"); plt.legend(); plt.tight_layout()
plt.yscale("log")
plt.savefig("learning_rate.png")

# ---- PPL 曲線（評価のみ）
plt.figure(figsize=(8,4))
ppl = np.exp(eval_["eval_loss"])
plt.plot(eval_["step"], ppl, label="eval_ppl", color="red")
plt.xlabel("step"); plt.ylabel("perplexity"); plt.legend(); plt.tight_layout()
plt.savefig("eval_ppl.png")