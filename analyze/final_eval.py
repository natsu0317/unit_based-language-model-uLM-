# fixed_acoustic_evaluation.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM
import sys
import os
from typing import Dict, List, Tuple
import random

sys.path.append('/home/mine/ulm')
from model import build_model
from peft import PeftModel

class AcousticUnitTokenizer:
    """音響ユニット専用トークナイザー（修正版）"""
    def __init__(self, vocab_size=129, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
    
    def __call__(self, text, return_tensors=None):
        """
        音響ユニットIDを直接処理
        input: "17 2 107 44 27 59" -> [17, 2, 107, 44, 27, 59]
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        all_ids = []
        for t in texts:
            try:
                # スペース区切りの数値を直接パース
                ids = [int(x) for x in t.strip().split() if x.strip()]
                # 語彙範囲チェック（0-128）
                ids = [id for id in ids if 0 <= id < self.vocab_size]
                all_ids.append(ids)
            except ValueError as e:
                print(f"Warning: Failed to parse '{t}': {e}")
                all_ids.append([])
        
        if return_tensors == "pt":
            if not all_ids or all(len(ids) == 0 for ids in all_ids):
                return {
                    'input_ids': torch.tensor([[self.pad_token_id]], dtype=torch.long),
                    'attention_mask': torch.tensor([[0]], dtype=torch.long)
                }
            
            # パディング処理
            max_len = max(len(ids) for ids in all_ids if len(ids) > 0)
            padded_ids = []
            attention_masks = []
            
            for ids in all_ids:
                if len(ids) == 0:
                    padded = [self.pad_token_id] * max_len
                    mask = [0] * max_len
                else:
                    padded = ids + [self.pad_token_id] * (max_len - len(ids))
                    mask = [1] * len(ids) + [0] * (max_len - len(ids))
                
                padded_ids.append(padded)
                attention_masks.append(mask)
            
            return {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        
        return all_ids

class FixedPerplexityEvaluator:
    """修正されたPerplexity評価器"""
    
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def compute_perplexity_correct(self, model, tokenizer, sequences: List[str], max_samples: int = 100) -> Dict:
        """正しいPerplexity計算"""
        print(f"\n📊 Computing Correct Perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        valid_samples = 0
        sample_perplexities = []
        
        print("🔍 Debugging tokenizer and model compatibility...")
        
        for i, sequence in enumerate(sequences[:max_samples]):
            try:
                # 1. 正しいトークン化
                inputs = tokenizer(sequence, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # デバッグ情報
                if i < 3:
                    print(f"\nSample {i}:")
                    print(f"  Original: {sequence[:50]}...")
                    print(f"  Input IDs: {inputs['input_ids'][0][:10].tolist()}...")
                    print(f"  Input shape: {inputs['input_ids'].shape}")
                    print(f"  ID range: {inputs['input_ids'].min().item()} - {inputs['input_ids'].max().item()}")
                    print(f"  Attention mask: {inputs['attention_mask'][0][:10].tolist()}...")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                    
                    if i < 3:
                        print(f"  Logits shape: {logits.shape}")
                        print(f"  Model vocab size: {logits.size(-1)}")
                        print(f"  Logits range: {logits.min().item():.2f} - {logits.max().item():.2f}")
                
                # 2. 正しいLoss計算（パディングを除外）
                shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, V)
                shift_labels = inputs['input_ids'][..., 1:].contiguous()  # (B, L-1)
                shift_attention = inputs['attention_mask'][..., 1:].contiguous()  # (B, L-1)
                
                # CrossEntropyLoss with ignore_index
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='none')
                losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                losses = losses.view(shift_labels.shape)  # (B, L-1)
                
                # マスクを適用（パディング部分を除外）
                valid_mask = (shift_labels != self.pad_token_id) & (shift_attention == 1)
                masked_losses = losses * valid_mask.float()
                
                # サンプルごとのPerplexity計算
                sample_loss_sum = masked_losses.sum(dim=1)  # (B,)
                sample_token_count = valid_mask.sum(dim=1).float()  # (B,)
                
                for j in range(sample_loss_sum.size(0)):
                    if sample_token_count[j] > 0:
                        sample_avg_loss = sample_loss_sum[j] / sample_token_count[j]
                        sample_ppl = torch.exp(sample_avg_loss).item()
                        
                        # 異常値チェック
                        if sample_ppl > self.pad_token_id * 2:  # vocab_size * 2 = 258を超える場合
                            print(f"⚠️ High perplexity detected: {sample_ppl:.2f} for sample {i}")
                            print(f"   Avg loss: {sample_avg_loss.item():.4f}")
                            print(f"   Token count: {sample_token_count[j].item()}")
                            
                            # より詳細なデバッグ
                            probs = torch.softmax(shift_logits[j], dim=-1)
                            min_prob = probs[valid_mask[j]].min().item()
                            max_prob = probs[valid_mask[j]].max().item()
                            print(f"   Prob range: {min_prob:.6f} - {max_prob:.6f}")
                        
                        sample_perplexities.append(sample_ppl)
                        total_loss += sample_loss_sum[j].item()
                        total_tokens += sample_token_count[j].item()
                        valid_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            avg_perplexity = np.exp(avg_loss)
            
            print(f"\n📊 Corrected Perplexity Statistics:")
            print(f"  Total valid tokens: {total_tokens}")
            print(f"  Valid samples: {valid_samples}")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Average PPL: {avg_perplexity:.2f}")
            print(f"  Median PPL: {np.median(sample_perplexities):.2f}")
            print(f"  Min PPL: {min(sample_perplexities):.2f}")
            print(f"  Max PPL: {max(sample_perplexities):.2f}")
            print(f"  Theoretical max PPL (vocab_size): {self.pad_token_id + 128}")  # 129
            
            return {
                'avg_perplexity': avg_perplexity,
                'median_perplexity': np.median(sample_perplexities),
                'min_perplexity': min(sample_perplexities),
                'max_perplexity': max(sample_perplexities),
                'valid_samples': valid_samples,
                'valid_tokens': total_tokens,
                'avg_loss': avg_loss
            }
        else:
            return {
                'avg_perplexity': float('inf'), 
                'valid_samples': 0,
                'valid_tokens': 0
            }

class SWuggyEvaluator:
    """sWUGGY評価器（修正版）"""
    
    def __init__(self, vocab_size=129, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
    def create_swuggy_pairs(self, sequences: List[str], n_pairs: int = 200) -> List[Tuple[str, str, int]]:
        """sWUGGYペアを作成"""
        pairs = []
        
        for _ in range(n_pairs):
            real_seq = random.choice(sequences)
            fake_seq = self._create_phonologically_invalid_sequence(real_seq)
            
            if random.random() < 0.5:
                pairs.append((real_seq, fake_seq, 1))  # real が先
            else:
                pairs.append((fake_seq, real_seq, 0))  # fake が先
                
        return pairs
    
    def _create_phonologically_invalid_sequence(self, real_seq: str) -> str:
        """音韻的に無効なシーケンスを生成"""
        units = real_seq.split()
        
        if random.random() < 0.5:
            # ランダムシャッフル
            shuffled = units.copy()
            random.shuffle(shuffled)
            return ' '.join(shuffled)
        else:
            # ランダム置換
            modified = units.copy()
            n_replace = max(1, len(units) // 4)
            positions = random.sample(range(len(units)), n_replace)
            
            for pos in positions:
                modified[pos] = str(random.randint(1, self.vocab_size-1))
            
            return ' '.join(modified)
    
    def evaluate(self, model, tokenizer, test_sequences: List[str], n_pairs: int = 200) -> Dict:
        """sWUGGY評価を実行"""
        print(f"Creating {n_pairs} sWUGGY test pairs...")
        pairs = self.create_swuggy_pairs(test_sequences, n_pairs)
        
        correct = 0
        total_log_prob_diff = 0
        
        print("Running sWUGGY evaluation...")
        for i, (seq1, seq2, correct_label) in enumerate(pairs):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(pairs)}")
            
            prob1 = self._compute_sequence_log_probability(model, tokenizer, seq1)
            prob2 = self._compute_sequence_log_probability(model, tokenizer, seq2)
            
            predicted_label = 1 if prob1 > prob2 else 0
            
            if predicted_label == correct_label:
                correct += 1
            
            total_log_prob_diff += abs(prob1 - prob2)
        
        accuracy = correct / len(pairs)
        avg_log_prob_diff = total_log_prob_diff / len(pairs)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(pairs),
            'avg_log_prob_diff': avg_log_prob_diff
        }
    
    def _compute_sequence_log_probability(self, model, tokenizer, sequence: str) -> float:
        """シーケンスの対数確率を計算（修正版）"""
        try:
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 正しい対数確率計算
                log_probs = torch.log_softmax(logits, dim=-1)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                sequence_log_prob = 0
                valid_tokens = 0
                
                for i in range(input_ids.size(1) - 1):
                    if attention_mask[0, i+1] == 1:  # パディングでない場合のみ
                        next_token_id = input_ids[0, i + 1].item()
                        if next_token_id != self.pad_token_id:
                            token_log_prob = log_probs[0, i, next_token_id].item()
                            sequence_log_prob += token_log_prob
                            valid_tokens += 1
                
                # 長さで正規化
                return sequence_log_prob / max(valid_tokens, 1)
                
        except Exception as e:
            return float('-inf')

class SBlimpEvaluator:
    """sBLIMP評価器（修正版）"""
    
    def __init__(self, vocab_size=129, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
    
    def create_sblimp_pairs(self, sequences: List[str], n_pairs: int = 200) -> List[Tuple[str, str, int]]:
        """sBLIMPペアを作成"""
        pairs = []
        
        for _ in range(n_pairs):
            grammatical_seq = random.choice(sequences)
            ungrammatical_seq = self._create_syntactically_invalid_sequence(grammatical_seq)
            
            if random.random() < 0.5:
                pairs.append((grammatical_seq, ungrammatical_seq, 1))
            else:
                pairs.append((ungrammatical_seq, grammatical_seq, 0))
                
        return pairs
    
    def _create_syntactically_invalid_sequence(self, grammatical_seq: str) -> str:
        """構文的に無効なシーケンスを生成"""
        units = grammatical_seq.split()
        
        if len(units) < 3:
            return grammatical_seq
        
        if random.random() < 0.4:
            # 部分的逆順
            start = random.randint(0, len(units) - 3)
            end = min(start + random.randint(2, 4), len(units))
            units[start:end] = units[start:end][::-1]
            return ' '.join(units)
        elif random.random() < 0.7:
            # 重複挿入
            pos = random.randint(1, len(units) - 1)
            duplicate_unit = units[pos]
            units.insert(pos, duplicate_unit)
            return ' '.join(units)
        else:
            # ランダム削除
            if len(units) > 3:
                del_pos = random.randint(1, len(units) - 2)
                del units[del_pos]
            return ' '.join(units)
    
    def evaluate(self, model, tokenizer, test_sequences: List[str], n_pairs: int = 200) -> Dict:
        """sBLIMP評価を実行"""
        print(f"Creating {n_pairs} sBLIMP test pairs...")
        pairs = self.create_sblimp_pairs(test_sequences, n_pairs)
        
        correct = 0
        total_log_prob_diff = 0
        
        print("Running sBLIMP evaluation...")
        for i, (seq1, seq2, correct_label) in enumerate(pairs):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(pairs)}")
            
            prob1 = self._compute_sequence_log_probability(model, tokenizer, seq1)
            prob2 = self._compute_sequence_log_probability(model, tokenizer, seq2)
            
            predicted_label = 1 if prob1 > prob2 else 0
            
            if predicted_label == correct_label:
                correct += 1
            
            total_log_prob_diff += abs(prob1 - prob2)
        
        accuracy = correct / len(pairs)
        avg_log_prob_diff = total_log_prob_diff / len(pairs)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(pairs),
            'avg_log_prob_diff': avg_log_prob_diff
        }
    
    def _compute_sequence_log_probability(self, model, tokenizer, sequence: str) -> float:
        """シーケンスの対数確率を計算（修正版）"""
        try:
            inputs = tokenizer(sequence, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                log_probs = torch.log_softmax(logits, dim=-1)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                sequence_log_prob = 0
                valid_tokens = 0
                
                for i in range(input_ids.size(1) - 1):
                    if attention_mask[0, i+1] == 1:
                        next_token_id = input_ids[0, i + 1].item()
                        if next_token_id != self.pad_token_id:
                            token_log_prob = log_probs[0, i, next_token_id].item()
                            sequence_log_prob += token_log_prob
                            valid_tokens += 1
                
                return sequence_log_prob / max(valid_tokens, 1)
                
        except Exception as e:
            return float('-inf')

class CorrectedSpeechEvaluator:
    """修正された音声言語モデル評価器"""
    
    def __init__(self, vocab_size=129, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.tokenizer = AcousticUnitTokenizer(vocab_size, pad_token_id)
        self.perplexity_evaluator = FixedPerplexityEvaluator(pad_token_id)
        self.swuggy_evaluator = SWuggyEvaluator(vocab_size, pad_token_id)
        self.sblimp_evaluator = SBlimpEvaluator(vocab_size, pad_token_id)
        
    def evaluate_model(self, model, model_name: str, test_sequences: List[str]) -> Dict:
        """単一モデルの包括評価（修正版）"""
        print(f"\n🔍 Evaluating {model_name} (CORRECTED)...")
        
        results = {}
        
        # 1. 修正されたPerplexity評価
        print("1. Corrected Perplexity Evaluation")
        ppl_results = self.perplexity_evaluator.compute_perplexity_correct(model, self.tokenizer, test_sequences, max_samples=100)
        results['perplexity'] = ppl_results
        print(f"Corrected Average Perplexity: {ppl_results['avg_perplexity']:.2f}")
        
        # 2. sWUGGY評価
        print("2. sWUGGY Evaluation")
        swuggy_results = self.swuggy_evaluator.evaluate(model, self.tokenizer, test_sequences, n_pairs=200)
        results['swuggy'] = swuggy_results
        print(f"sWUGGY Accuracy: {swuggy_results['accuracy']:.3f} ({swuggy_results['correct']}/{swuggy_results['total']})")
        
        # 3. sBLIMP評価
        print("3. sBLIMP Evaluation")
        sblimp_results = self.sblimp_evaluator.evaluate(model, self.tokenizer, test_sequences, n_pairs=200)
        results['sblimp'] = sblimp_results
        print(f"sBLIMP Accuracy: {sblimp_results['accuracy']:.3f} ({sblimp_results['correct']}/{sblimp_results['total']})")
        
        return results
    
    def compare_models(self, models: Dict, test_sequences: List[str]) -> Dict:
        """複数モデルの比較評価"""
        all_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            results = self.evaluate_model(model, model_name, test_sequences)
            all_results[model_name] = results
        
        return all_results
    
    def create_corrected_visualization(self, results: Dict, save_path: str = "corrected_speech_evaluation.png"):
        """修正された結果の可視化"""
        print("\n📈 Creating corrected visualization...")
        
        models = list(results.keys())
        swuggy_scores = [results[model]['swuggy']['accuracy'] for model in models]
        sblimp_scores = [results[model]['sblimp']['accuracy'] for model in models]
        perplexities = [results[model]['perplexity']['avg_perplexity'] for model in models]
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Corrected Speech Language Model Evaluation', fontsize=16, fontweight='bold')
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # 1. Perplexity比較（修正版）
        ax1 = axes[0, 0]
        bars = ax1.bar(models, perplexities, color=colors[:len(models)], alpha=0.8)
        ax1.set_ylabel('Perplexity (lower is better)')
        ax1.set_title('Corrected Perplexity Comparison')
        ax1.grid(True, alpha=0.3)
        
        # 理論的最大値と現実的範囲を表示
        ax1.axhline(y=129, color='red', linestyle='--', alpha=0.7, label='Theoretical Max (129)')
        ax1.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='Good Performance (~50)')
        ax1.legend()
        
        for bar, ppl in zip(bars, perplexities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{ppl:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. sWUGGY vs sBLIMP
        ax2 = axes[0, 1]
        for i, model in enumerate(models):
            ax2.scatter(swuggy_scores[i], sblimp_scores[i], 
                       s=200, color=colors[i], alpha=0.7, label=model)
            ax2.annotate(model, (swuggy_scores[i], sblimp_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('sWUGGY Accuracy')
        ax2.set_ylabel('sBLIMP Accuracy')
        ax2.set_title('Phonological vs Syntactic Performance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 総合スコア
        ax3 = axes[1, 0]
        # Perplexityを逆転正規化（低い方が良い）
        max_ppl = max(perplexities)
        normalized_ppl = [(max_ppl - ppl) / max_ppl for ppl in perplexities]
        
        metrics = ['sWUGGY', 'sBLIMP', 'Perplexity\n(Normalized)']
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model in enumerate(models):
            scores = [swuggy_scores[i], sblimp_scores[i], normalized_ppl[i]]
            ax3.bar(x + i * width, scores, width, label=model, color=colors[i], alpha=0.8)
        
        ax3.set_xlabel('Evaluation Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Overall Performance (Corrected)')
        ax3.set_xticks(x + width / 2)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 詳細統計表
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        for model in models:
            r = results[model]
            table_data.append([
                model,
                f"{r['swuggy']['accuracy']:.3f}",
                f"{r['sblimp']['accuracy']:.3f}",
                f"{r['perplexity']['avg_perplexity']:.1f}",
                f"{r['perplexity']['valid_tokens']}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'sWUGGY', 'sBLIMP', 'PPL (Fixed)', 'Valid Tokens'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Corrected Results Summary')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Corrected visualization saved to {save_path}")
        
        return fig

def main():
    print("=== CORRECTED SPEECH LANGUAGE MODEL EVALUATION ===")
    print("🔧 Using proper acoustic unit tokenizer and fixed perplexity calculation")
    
    # テストデータ準備
    print("\n📚 Loading test data...")
    dev_data = pd.read_csv('../units_dev.csv')
    
    # フェアなテストデータを作成
    np.random.seed(42)
    test_indices = np.random.choice(len(dev_data), size=min(200, len(dev_data)//2), replace=False)
    test_data = dev_data.iloc[test_indices].reset_index(drop=True)
    test_sequences = test_data['text'].tolist()
    
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Sample sequence: {test_sequences[0][:50]}...")
    
    # モデル準備
    print("\n🤖 Loading models...")
    models = {}
    
    # 1. GPT-2 Scratch
    gpt2_path = "../gpt2-unit-lm/ulm-gpt2-scratch-final"
    try:
        print("📦 Loading GPT-2 scratch model...")
        gpt2_model = AutoModelForCausalLM.from_pretrained(
            gpt2_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        
        print(f"GPT-2 model vocab size: {gpt2_model.config.vocab_size}")
        models['GPT-2 Scratch'] = {'model': gpt2_model}
        print("✅ GPT-2 loaded")
    except Exception as e:
        print(f"❌ Failed to load GPT-2: {e}")
    
    # 2. Qwen Transfer
    lora_path = "../ulm-qwen3b-merged"
    try:
        print("📦 Loading Qwen transfer model...")
        base_model = build_model()
        model = PeftModel.from_pretrained(base_model, lora_path)
        qwen_model = model.merge_and_unload()
        
        print(f"Qwen model vocab size: {qwen_model.config.vocab_size}")
        models['Qwen Transfer'] = {'model': qwen_model}
        print("✅ Qwen loaded")
        
        del base_model, model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Failed to load Qwen: {e}")
    
    if not models:
        print("❌ No models loaded!")
        return
    
    # 修正された評価実行
    evaluator = CorrectedSpeechEvaluator(vocab_size=129, pad_token_id=0)
    all_results = evaluator.compare_models(models, test_sequences)
    
    # 可視化
    evaluator.create_corrected_visualization(all_results)
    
    # 結果サマリー
    print("\n" + "="*60)
    print("🎉 CORRECTED EVALUATION COMPLETED!")
    print("="*60)
    
    for model_name, results in all_results.items():
        print(f"\n📊 {model_name} (CORRECTED):")
        print(f"  sWUGGY: {results['swuggy']['accuracy']:.3f}")
        print(f"  sBLIMP: {results['sblimp']['accuracy']:.3f}")
        print(f"  Perplexity: {results['perplexity']['avg_perplexity']:.2f}")
        print(f"  Valid tokens: {results['perplexity']['valid_tokens']}")
    
    # 比較分析
    if len(all_results) >= 2:
        models_list = list(all_results.keys())
        model1, model2 = models_list[0], models_list[1]
        r1, r2 = all_results[model1], all_results[model2]
        
        print(f"\n🏆 CORRECTED COMPARISON:")
        print(f"  Perplexity: {model1} {r1['perplexity']['avg_perplexity']:.2f} vs {model2} {r2['perplexity']['avg_perplexity']:.2f}")
        
        ppl_winner = model1 if r1['perplexity']['avg_perplexity'] < r2['perplexity']['avg_perplexity'] else model2
        print(f"  🏅 Perplexity Winner: {ppl_winner}")
        
        # 理論的妥当性チェック
        for model_name, results in all_results.items():
            ppl = results['perplexity']['avg_perplexity']
            if ppl > 129:
                print(f"⚠️ {model_name}: PPL {ppl:.2f} > 129 (still abnormal)")
            elif ppl < 10:
                print(f"⚠️ {model_name}: PPL {ppl:.2f} < 10 (suspiciously low)")
            else:
                print(f"✅ {model_name}: PPL {ppl:.2f} (reasonable range)")

if __name__ == "__main__":
    main()