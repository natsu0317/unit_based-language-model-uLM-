import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os
import numpy as np

def load_trained_model():
    model = GPT2LMHeadModel.from_pretrained("../unit-lm-final")
    model.eval()  # 評価モード
        
    print(f"✅ モデル読み込み成功")
    print(f"  パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  語彙サイズ: {model.config.vocab_size}")
    print(f"  埋め込み次元: {model.config.n_embd}")
    print(f"  層数: {model.config.n_layer}")

    return model

def model_test(model):
    sample_input = torch.tensor([[15, 42, 67, 23]])  # バッチサイズ1
    
    print(f"入力: {sample_input}")
    print(f"入力形状: {sample_input.shape}")

    with torch.no_grad(): # 学習無し
        outputs = model(sample_input)
        logits = outputs.logits # 予測値
        print(f"出力形状: {logits.shape}")
        print(f"最後の位置の予測分布形状: {logits[0, -1, :].shape}")
        
        next_token_logits = logits[0,-1, :]
        # もっとも予測値の高いindex
        predicted_token = torch.argmax(next_token_logits).item()
        # 予測のconfidence(logits->確率)
        confidence = torch.softmax(next_token_logits, dim=-1)[predicted_token].item()
        
        print(f"\\n予測結果:")
        print(f"  入力系列: [15, 42, 67, 23]")
        print(f"  予測次トークン: {predicted_token}")
        print(f"  予測確信度: {confidence:.4f}")
        
        # 上位5個の予測
        top5_probs, top5_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), 5)
        print(f"\\n上位5予測:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            print(f"    {i+1}. トークン{idx.item()}: {prob.item():.4f}")


if __name__ == "__main__":
    model = load_trained_model()
    model_test(model)