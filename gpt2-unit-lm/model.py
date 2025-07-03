# model.py
import torch
from transformers import GPT2Config, GPT2LMHeadModel

VOCAB = 129   # PAD0 + 128 unit

def build_model():
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_positions=1024,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=2048,
        pad_token_id=0,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(cfg)