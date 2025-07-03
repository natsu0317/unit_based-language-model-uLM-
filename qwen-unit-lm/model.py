# model.py
import copy, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

VOCAB_SIZE = 129            # 128 unit + pad

def build_model(
        base_name: str = "Qwen/Qwen2.5-3B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
    print("🔻  Loading base model …")
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    cfg = copy.deepcopy(base.config)
    cfg.vocab_size   = VOCAB_SIZE # 15万語->129
    cfg.pad_token_id = 0

    print("🔻  Building new model …")
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).to("cuda")

    # 同形状だけコピー(embedding, lm_head以外)
    copied = 0
    for n, p in base.state_dict().items():
        if n in model.state_dict() and p.shape == model.state_dict()[n].shape:
            model.state_dict()[n].copy_(p)
            copied += 1
    print(f"✅  copied {copied} tensors")

    # 埋め込み / LM ヘッドを Xavier 初期化
    torch.nn.init.xavier_uniform_(model.get_input_embeddings().weight)
    torch.nn.init.xavier_uniform_(model.lm_head.weight)

    # LoRA
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 参照モデルを解放
    del base
    torch.cuda.empty_cache()
    return model