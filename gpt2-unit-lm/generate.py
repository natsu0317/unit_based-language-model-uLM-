import torch
from transformers import GPT2LMHeadModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT2LMHeadModel.from_pretrained(
    "./ulm-gpt2-scratch-final",
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
).to(DEVICE)

prompt = torch.tensor([[1, 20, 11, 7]], device=DEVICE)
out = model.generate(prompt, max_length=30, top_p=0.9, do_sample=True)
print(out[0].tolist())