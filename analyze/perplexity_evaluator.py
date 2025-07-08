# === Perplexity_evaluator.py ===
import torch
import numpy as np
from tqdm import tqdm

class PerplexityEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, tokenizer, test_data, max_samples=500):
        """モデルの複雑性評価"""
        print(f"Running Perplexity evaluation on {max_samples} samples...")
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        sequence_perplexities = []
        valid_sequences = 0
        
        with torch.no_grad():
            for i, text in enumerate(tqdm(test_data['text'][:max_samples])):
                try:
                    unit_ids = [int(x) + 1 for x in text.split()]
                    if len(unit_ids) < 2:
                        continue
                    
                    # 入力とラベルを準備
                    input_ids = torch.tensor([unit_ids[:-1]], dtype=torch.long)
                    labels = torch.tensor([unit_ids[1:]], dtype=torch.long)
                    
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        labels = labels.cuda()
                    
                    # モデル実行
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss.item()
                    
                    # シーケンス単位の統計
                    sequence_length = len(unit_ids) - 1
                    sequence_perplexity = np.exp(loss)
                    sequence_perplexities.append(sequence_perplexity)
                    
                    # 全体統計
                    total_loss += loss * sequence_length
                    total_tokens += sequence_length
                    valid_sequences += 1
                    
                except Exception as e:
                    continue
        
        # 結果計算
        overall_perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        avg_sequence_perplexity = np.mean(sequence_perplexities) if sequence_perplexities else float('inf')
        median_perplexity = np.median(sequence_perplexities) if sequence_perplexities else float('inf')
        std_perplexity = np.std(sequence_perplexities) if sequence_perplexities else 0
        
        results = {
            'overall_perplexity': overall_perplexity,
            'average_sequence_perplexity': avg_sequence_perplexity,
            'median_perplexity': median_perplexity,
            'std_perplexity': std_perplexity,
            'valid_sequences': valid_sequences,
            'total_tokens': total_tokens,
            'sequence_perplexities': sequence_perplexities
        }
        
        print(f"Overall Perplexity: {overall_perplexity:.2f}")
        print(f"Average Sequence Perplexity: {avg_sequence_perplexity:.2f}")
        print(f"Median Perplexity: {median_perplexity:.2f}")
        print(f"Valid sequences: {valid_sequences}")
        
        return results