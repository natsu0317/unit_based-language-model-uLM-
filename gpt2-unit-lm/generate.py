# generate.py
import torch
from transformers import AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
            "unit-lm-scratch-final", torch_dtype=torch.bfloat16
        ).to(DEVICE)
model.eval()

prompt = torch.tensor([[1, 20, 11, 7]], device=DEVICE)   # 1-128
with torch.no_grad():
    out = model.generate(
        prompt, max_length=30, do_sample=True, top_p=0.9, temperature=1.0
    )
print("generated IDs:", out[0].tolist())
print("original clusters:", [x-1 for x in out[0].tolist()])