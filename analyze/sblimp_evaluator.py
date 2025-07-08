# === sblimp_evaluator.py ===
import torch
import numpy as np
import random
from tqdm import tqdm

class SBLiMPEvaluator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.results = {}
    
    def create_sblimp_pairs(self, unit_data, n_pairs=200):
        """SBLiMP風のテストペア作成"""
        print(f"Creating {n_pairs} SBLiMP test pairs...")
        
        pairs = []
        sequences = unit_data['text'].tolist()
        
        for i in range(n_pairs):
            # 文法的に正しいシーケンス
            grammatical_seq = sequences[i % len(sequences)]
            
            # 文法的に間違ったシーケンス作成
            ungrammatical_seq = self._create_ungrammatical_sequence(
                grammatical_seq, violation_type='structural'
            )
            
            pairs.append({
                'id': i,
                'grammatical': grammatical_seq,
                'ungrammatical': ungrammatical_seq,
                'violation_type': 'structural',
                'gram_length': len(grammatical_seq.split()),
                'ungram_length': len(ungrammatical_seq.split())
            })
        
        return pairs
    
    def _create_ungrammatical_sequence(self, grammatical_seq, violation_type='structural'):
        """非文法的シーケンス作成"""
        units = grammatical_seq.split()
        
        if violation_type == 'structural':
            # 構造的違反：順序の破壊
            if len(units) >= 4:
                # 中間部分をランダムに並び替え
                mid_start = len(units) // 4
                mid_end = 3 * len(units) // 4
                middle_part = units[mid_start:mid_end]
                random.shuffle(middle_part)
                
                ungrammatical = units[:mid_start] + middle_part + units[mid_end:]
            else:
                # 短いシーケンスは完全シャッフル
                ungrammatical = units.copy()
                random.shuffle(ungrammatical)
                
        elif violation_type == 'repetition':
            # 反復違反：不自然な反復
            if len(units) >= 3:
                repeat_pos = random.randint(1, len(units) - 2)
                repeat_unit = units[repeat_pos]
                # 同じ単位を3回連続で挿入
                ungrammatical = (units[:repeat_pos] + 
                               [repeat_unit, repeat_unit, repeat_unit] + 
                               units[repeat_pos + 1:])
            else:
                ungrammatical = units + units  # 全体を反復
                
        elif violation_type == 'deletion':
            # 削除違反：重要な部分を削除
            if len(units) >= 3:
                # ランダムに30%の単位を削除
                keep_indices = random.sample(range(len(units)), 
                                           max(1, int(len(units) * 0.7)))
                keep_indices.sort()
                ungrammatical = [units[i] for i in keep_indices]
            else:
                ungrammatical = units[:1]  # 最初の単位のみ残す
        
        return " ".join(ungrammatical)
    
    def evaluate_model(self, model, tokenizer, test_pairs):
        """モデルのSBLiMP評価"""
        print("Running SBLiMP evaluation...")
        
        model.eval()
        correct = 0
        total = len(test_pairs)
        detailed_results = []
        
        with torch.no_grad():
            for pair in tqdm(test_pairs):
                gram_seq = pair['grammatical']
                ungram_seq = pair['ungrammatical']
                
                # 各シーケンスの対数尤度計算
                gram_logprob = self._compute_log_probability(model, tokenizer, gram_seq)
                ungram_logprob = self._compute_log_probability(model, tokenizer, ungram_seq)
                
                # 文法的シーケンスの方が高確率なら正解
                is_correct = gram_logprob > ungram_logprob
                if is_correct:
                    correct += 1
                
                detailed_results.append({
                    'pair_id': pair['id'],
                    'grammatical_logprob': gram_logprob,
                    'ungrammatical_logprob': ungram_logprob,
                    'difference': gram_logprob - ungram_logprob,
                    'correct': is_correct,
                    'violation_type': pair['violation_type']
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
        
        print(f"SBLiMP Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"Average log probability difference: {avg_difference:.3f}")
        
        return results
    
    def _compute_log_probability(self, model, tokenizer, sequence):
        """シーケンスの対数確率計算（Swuggyと同じ）"""
        try:
            unit_ids = [int(x) + 1 for x in sequence.split()]
            
            if len(unit_ids) < 2:
                return float('-inf')
            
            input_ids = torch.tensor([unit_ids], dtype=torch.long)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            log_prob = -loss * len(unit_ids)
            return log_prob
            
        except Exception as e:
            return float('-inf')