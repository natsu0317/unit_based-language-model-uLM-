import torch
from transformers import AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
           "ulm-gpt2-scratch-final", torch_dtype=torch.bfloat16
        ).to(DEVICE)
model.eval()

prompt = torch.tensor([[1, 20, 11, 7]], device=DEVICE)
out = model.generate(prompt, max_length=30, do_sample=True, top_p=0.9)
print("generated IDs:", out[0].tolist())