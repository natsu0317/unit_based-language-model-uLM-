# generate.py
import torch
from transformers import AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "ulm-qwen3b-merged", torch_dtype=torch.bfloat16
).to(DEVICE)
model.eval()

# 例: 4 トークン与えて続き 30 トークン生成
prompt = torch.tensor([[1, 20, 11, 7]], device=DEVICE)  # 1-128 が unit
with torch.no_grad():
    out = model.generate(
        prompt,
        max_length=30,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )
print("generated ids:", out[0].tolist())
