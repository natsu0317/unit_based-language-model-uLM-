import torch
from transformers import GPT2Config, GPT2LMHeadModel

VOCAB = 129             # PAD0 + 128 unit

def build_model():
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_positions=1024,
        n_embd=1024,     # ★ Qwen と同じ隠れ次元
        n_layer=12,      # ★ 12 層（A100 1 枚で収まる最大級）
        n_head=16,
        n_inner=4096,
        pad_token_id=0,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(cfg)