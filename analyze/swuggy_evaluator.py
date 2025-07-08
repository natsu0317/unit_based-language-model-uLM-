# === swuggy_evaluator.py ===
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import random

class SwuggyEvaluator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.results = {}
    
    def create_swuggy_pairs(self, unit_data, n_pairs=200):
        """Swuggy風のテストペア作成"""
        print(f"Creating {n_pairs} Swuggy test pairs...")
        
        pairs = []
        sequences = unit_data['text'].tolist()
        
        for i in range(n_pairs):
            # 実際のシーケンス選択
            real_seq = sequences[i % len(sequences)]
            
            # 疑似シーケンス作成（複数の方法）
            pseudo_seq = self._create_pseudo_sequence(real_seq, method='phonological')
            
            pairs.append({
                'id': i,
                'real_sequence': real_seq,
                'pseudo_sequence': pseudo_seq,
                'real_length': len(real_seq.split()),
                'pseudo_length': len(pseudo_seq.split())
            })
        
        return pairs
    
    def _create_pseudo_sequence(self, real_sequence, method='phonological'):
        """疑似シーケンス作成"""
        units = real_sequence.split()
        
        if method == 'phonological':
            # 音韻的に不自然な置換
            pseudo_units = []
            for i, unit in enumerate(units):
                if random.random() < 0.25:  # 25%の確率で置換
                    # 音韻的に遠い単位に置換
                    current_unit = int(unit)
                    # 大きく離れた値に置換（音韻的に不自然）
                    if current_unit < 64:
                        pseudo_unit = str(random.randint(90, 127))
                    else:
                        pseudo_unit = str(random.randint(0, 37))
                    pseudo_units.append(pseudo_unit)
                else:
                    pseudo_units.append(unit)
            
        elif method == 'random':
            # 完全ランダム置換
            pseudo_units = [str(random.randint(0, 127)) for _ in units]
            
        elif method == 'shuffle':
            # シャッフル
            pseudo_units = units.copy()
            random.shuffle(pseudo_units)
        
        return " ".join(pseudo_units)
    
    def evaluate_model(self, model, tokenizer, test_pairs):
        """モデルのSwuggy評価"""
        print("Running Swuggy evaluation...")
        
        model.eval()
        correct = 0
        total = len(test_pairs)
        detailed_results = []
        
        with torch.no_grad():
            for pair in tqdm(test_pairs):
                real_seq = pair['real_sequence']
                pseudo_seq = pair['pseudo_sequence']
                
                # 各シーケンスの対数尤度計算
                real_logprob = self._compute_log_probability(model, tokenizer, real_seq)
                pseudo_logprob = self._compute_log_probability(model, tokenizer, pseudo_seq)
                
                # 実際のシーケンスの方が高確率なら正解
                is_correct = real_logprob > pseudo_logprob
                if is_correct:
                    correct += 1
                
                detailed_results.append({
                    'pair_id': pair['id'],
                    'real_logprob': real_logprob,
                    'pseudo_logprob': pseudo_logprob,
                    'difference': real_logprob - pseudo_logprob,
                    'correct': is_correct
                })
        
        accuracy = correct / total
        avg_difference = np.mean([r['difference'] for r in detailed_results])
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_log_difference': avg_difference,
            'detailed_results': detailed_results
        }
        
        print(f"Swuggy Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"Average log probability difference: {avg_difference:.3f}")
        
        return results
    
    def _compute_log_probability(self, model, tokenizer, sequence):
        """シーケンスの対数確率計算"""
        try:
            # 音響単位をトークンIDに変換
            unit_ids = [int(x) + 1 for x in sequence.split()]  # +1 shift for PAD=0
            
            if len(unit_ids) < 2:
                return float('-inf')
            
            input_ids = torch.tensor([unit_ids], dtype=torch.long)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # 対数尤度計算
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            
            # 負の対数尤度を対数確率に変換
            log_prob = -loss * len(unit_ids)
            return log_prob
            
        except Exception as e:
            print(f"Error computing log probability: {e}")
            return float('-inf')