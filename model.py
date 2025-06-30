from transformers import GPT2Config, GPT2LMHeadModel
import torch

def create_unit_language_model():
    config = GPT2Config(
        vocab_size=128,
        n_positions=512,
        n_embd=512, # 埋め込み次元
        n_layer=6, # Transformer層数
        n_head=8, # attentionhead数
        n_inner=2048,
        activation_function="gelu",
    )

    model = GPT2LMHeadModel(config)
    return model, config

def test_model():
    model,config = create_unit_language_model()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    test_input = torch.tensor([[15, 42, 67, 23, 89]], device=device) 
    with torch.no_grad():
        outputs = model(test_input)
        logits = outputs.logits
    last_logits = logits[0, -1, :]  # 最後の位置の予測
    probabilities = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, 5)
    
    print(f"\\n🎯 次の音響単位の予測（上位5個）:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"  {i+1}位: Unit {idx.item()} (確率: {prob.item():.4f})")
    
    print("\\n✅ モデルテスト完了!")

if __name__ == "__main__":
    test_model()