# === comprehensive_evaluator.py ===
from swuggy_evaluator import SwuggyEvaluator
from sblimp_evaluator import SBLiMPEvaluator  
from perplexity_evaluator import PerplexityEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ComprehensiveEvaluator:
    def __init__(self):
        self.swuggy_eval = SwuggyEvaluator()
        self.sblimp_eval = SBLiMPEvaluator()
        self.complexity_eval = PerplexityEvaluator()
        self.results = {}
    
    def evaluate_all_models(self, models, test_data):
        """全モデルの包括的評価"""
        print("=== COMPREHENSIVE MODEL EVALUATION ===")
        
        all_results = {}
        
        for model_name, model_info in models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            
            # 1. Swuggy評価
            print("1. Swuggy Evaluation")
            swuggy_pairs = self.swuggy_eval.create_swuggy_pairs(test_data, n_pairs=200)
            swuggy_results = self.swuggy_eval.evaluate_model(model, tokenizer, swuggy_pairs)
            
            # 2. SBLiMP評価
            print("2. SBLiMP Evaluation")
            sblimp_pairs = self.sblimp_eval.create_sblimp_pairs(test_data, n_pairs=200)
            sblimp_results = self.sblimp_eval.evaluate_model(model, tokenizer, sblimp_pairs)
            
            # 3. Complexity評価
            print("3. Complexity Evaluation")
            complexity_results = self.complexity_eval.evaluate_model(model, tokenizer, test_data)
            
            # 結果統合
            all_results[model_name] = {
                'swuggy': swuggy_results,
                'sblimp': sblimp_results,
                'complexity': complexity_results,
                'model_info': model_info
            }
            
            # 中間結果表示
            print(f"📊 {model_name} Results:")
            print(f"   Swuggy: {swuggy_results['accuracy']:.3f}")
            print(f"   SBLiMP: {sblimp_results['accuracy']:.3f}")
            print(f"   Complexity: {complexity_results['overall_perplexity']:.2f}")
        
        self.results = all_results
        return all_results
    
    def analyze_transfer_learning_effect(self):
        """転移学習効果の詳細分析"""
        print("\n=== TRANSFER LEARNING EFFECT ANALYSIS ===")
        
        if len(self.results) < 2:
            print("❌ Need at least 2 models for comparison")
            return None
        
        # モデル識別
        transfer_model = None
        scratch_model = None
        baseline_model = None
        
        for model_name, results in self.results.items():
            model_type = results['model_info'].get('type', '')
            
            if 'transfer' in model_type or 'finetuned' in model_name:
                transfer_model = (model_name, results)
            elif 'scratch' in model_type or 'gpt2' in model_name:
                scratch_model = (model_name, results)
            elif 'pretrained' in model_type or 'baseline' in model_name:
                baseline_model = (model_name, results)
        
        # 比較分析実行
        comparisons = {}
        
        if transfer_model and scratch_model:
            comparisons['transfer_vs_scratch'] = self._compare_two_models(
                transfer_model, scratch_model, "Transfer Learning vs Scratch"
            )
        
        if transfer_model and baseline_model:
            comparisons['transfer_vs_baseline'] = self._compare_two_models(
                transfer_model, baseline_model, "Transfer Learning vs Baseline"
            )
        
        if scratch_model and baseline_model:
            comparisons['scratch_vs_baseline'] = self._compare_two_models(
                scratch_model, baseline_model, "Scratch vs Baseline"
            )
        
        return comparisons
    
    def _compare_two_models(self, model1_data, model2_data, comparison_name):
        """2つのモデルの詳細比較"""
        name1, results1 = model1_data
        name2, results2 = model2_data
        
        print(f"\n📈 {comparison_name}")
        print(f"Comparing {name1} vs {name2}")
        
        # 各指標での比較
        swuggy1 = results1['swuggy']['accuracy']
        swuggy2 = results2['swuggy']['accuracy']
        swuggy_diff = swuggy1 - swuggy2
        swuggy_improvement = (swuggy_diff / max(swuggy1, swuggy2)) * 100
        
        sblimp1 = results1['sblimp']['accuracy']
        sblimp2 = results2['sblimp']['accuracy']
        sblimp_diff = sblimp1 - sblimp2
        sblimp_improvement = (sblimp_diff / max(sblimp1, sblimp2)) * 100
        
        complexity1 = results1['complexity']['overall_perplexity']
        complexity2 = results2['complexity']['overall_perplexity']
        complexity_diff = complexity2 - complexity1  # 低い方が良いので逆転
        complexity_improvement = (abs(complexity_diff) / max(complexity1, complexity2)) * 100
        
        comparison = {
            'models': {'model1': name1, 'model2': name2},
            'swuggy': {
                'model1_score': swuggy1,
                'model2_score': swuggy2,
                'difference': swuggy_diff,
                'improvement_pct': swuggy_improvement,
                'winner': name1 if swuggy_diff > 0 else name2
            },
            'sblimp': {
                'model1_score': sblimp1,
                'model2_score': sblimp2,
                'difference': sblimp_diff,
                'improvement_pct': sblimp_improvement,
                'winner': name1 if sblimp_diff > 0 else name2
            },
            'complexity': {
                'model1_score': complexity1,
                'model2_score': complexity2,
                'difference': complexity_diff,
                'improvement_pct': complexity_improvement,
                'winner': name1 if complexity_diff > 0 else name2
            }
        }
        
        # 総合勝者決定
        wins = [comparison['swuggy']['winner'], 
                comparison['sblimp']['winner'], 
                comparison['complexity']['winner']]
        
        if wins.count(name1) > wins.count(name2):
            comparison['overall_winner'] = name1
        elif wins.count(name2) > wins.count(name1):
            comparison['overall_winner'] = name2
        else:
            comparison['overall_winner'] = 'tie'
        
        # 結果表示
        print(f"  Swuggy: {name1} {swuggy1:.3f} vs {name2} {swuggy2:.3f} → {comparison['swuggy']['winner']}")
        print(f"  SBLiMP: {name1} {sblimp1:.3f} vs {name2} {sblimp2:.3f} → {comparison['sblimp']['winner']}")
        print(f"  Complexity: {name1} {complexity1:.2f} vs {name2} {complexity2:.2f} → {comparison['complexity']['winner']}")
        print(f"  Overall Winner: {comparison['overall_winner']}")
        
        return comparison
    
    def create_comprehensive_visualization(self):
        """包括的可視化作成"""
        if not self.results:
            print("No results to visualize")
            return None
        
        # データ準備
        model_names = list(self.results.keys())
        swuggy_scores = [self.results[name]['swuggy']['accuracy'] for name in model_names]
        sblimp_scores = [self.results[name]['sblimp']['accuracy'] for name in model_names]
        perplexities = [self.results[name]['complexity']['overall_perplexity'] for name in model_names]
        
        # 可視化作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Swuggy比較
        bars1 = axes[0, 0].bar(model_names, swuggy_scores, 
                              color=['lightblue', 'lightgreen', 'lightcoral'][:len(model_names)])
        axes[0, 0].set_ylabel('Swuggy Accuracy')
        axes[0, 0].set_title('Swuggy Performance Comparison')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars1, swuggy_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom')
        
        # 2. SBLiMP比較
        bars2 = axes[0, 1].bar(model_names, sblimp_scores,
                              color=['lightblue', 'lightgreen', 'lightcoral'][:len(model_names)])
        axes[0, 1].set_ylabel('SBLiMP Accuracy')
        axes[0, 1].set_title('SBLiMP Performance Comparison')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars2, sblimp_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Complexity比較
        bars3 = axes[0, 2].bar(model_names, perplexities,
                              color=['lightblue', 'lightgreen', 'lightcoral'][:len(model_names)])
        axes[0, 2].set_ylabel('Perplexity (lower is better)')
        axes[0, 2].set_title('Complexity Performance Comparison')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        for bar, perp in zip(bars3, perplexities):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities) * 0.01,
                          f'{perp:.1f}', ha='center', va='bottom')
        
        # 4. レーダーチャート（正規化スコア）
        if len(model_names) >= 2:
            # スコア正規化（0-1範囲）
            norm_swuggy = np.array(swuggy_scores)
            norm_sblimp = np.array(sblimp_scores)
            norm_complexity = 1 - (np.array(perplexities) / max(perplexities))  # 逆転して正規化
            
            angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
            angles += angles[:1]  # 円を閉じる
            
            ax_radar = plt.subplot(2, 3, 4, projection='polar')
            
            for i, name in enumerate(model_names):
                values = [norm_swuggy[i], norm_sblimp[i], norm_complexity[i]]
                values += values[:1]  # 円を閉じる
                
                ax_radar.plot(angles, values, 'o-', linewidth=2, label=name)
                ax_radar.fill(angles, values, alpha=0.25)
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(['Swuggy', 'SBLiMP', 'Complexity'])
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('Normalized Performance Radar')
            ax_radar.legend()
        
        # 5. 改善度比較（転移学習効果）
        comparisons = self.analyze_transfer_learning_effect()
        if comparisons and 'transfer_vs_scratch' in comparisons:
            comp = comparisons['transfer_vs_scratch']
            
            metrics = ['Swuggy', 'SBLiMP', 'Complexity']
            improvements = [
                comp['swuggy']['improvement_pct'],
                comp['sblimp']['improvement_pct'],
                comp['complexity']['improvement_pct']
            ]
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars5 = axes[1, 1].bar(metrics, improvements, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Improvement (%)')
            axes[1, 1].set_title('Transfer Learning Effect')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            for bar, imp in zip(bars5, improvements):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (1 if imp > 0 else -3),
                               f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
        
        # 6. 総合サマリー
        axes[1, 2].text(0.1, 0.9, 'Comprehensive Analysis Summary', 
                       fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
        
        summary_text = f"Models Evaluated: {len(model_names)}\n\n"
        
        # 各指標での最優秀モデル
        best_swuggy = model_names[np.argmax(swuggy_scores)]
        best_sblimp = model_names[np.argmax(sblimp_scores)]
        best_complexity = model_names[np.argmin(perplexities)]
        
        summary_text += f"Best Swuggy: {best_swuggy} ({max(swuggy_scores):.3f})\n"
        summary_text += f"Best SBLiMP: {best_sblimp} ({max(sblimp_scores):.3f})\n"
        summary_text += f"Best Complexity: {best_complexity} ({min(perplexities):.2f})\n\n"
        
        # 転移学習効果の判定
        if comparisons and 'transfer_vs_scratch' in comparisons:
            overall_winner = comparisons['transfer_vs_scratch']['overall_winner']
            if 'transfer' in overall_winner or 'finetuned' in overall_winner:
                summary_text += "✅ Transfer Learning Effective!\n"
            elif overall_winner == 'tie':
                summary_text += "⚖️ Mixed Results\n"
            else:
                summary_text += "❌ Transfer Learning Limited\n"
        
        axes[1, 2].text(0.1, 0.7, summary_text, fontsize=11, 
                       transform=axes[1, 2].transAxes, verticalalignment='top')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        # 空いているサブプロット
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_evaluation_results.png', dpi=200, bbox_inches='tight')
        print("Comprehensive evaluation visualization saved!")
        
        return fig
    
    def generate_comprehensive_report(self):
        """包括的分析レポート生成"""
        
        if not self.results:
            print("No results available for report generation")
            return None
        
        report = f"""# 転移学習効果の包括的評価結果

## 実験概要
- **評価指標**: Swuggy（音韻的妥当性）, SBLiMP（構文的理解）, Complexity（基本性能）
- **評価モデル数**: {len(self.results)}
- **テストデータ**: 音響単位シーケンス（units_dev.csv）

## 個別モデル性能

| モデル | Swuggy | SBLiMP | Complexity | 総合評価 |
|--------|--------|--------|------------|----------|
"""
        
        for model_name, results in self.results.items():
            swuggy_acc = results['swuggy']['accuracy']
            sblimp_acc = results['sblimp']['accuracy']
            complexity = results['complexity']['overall_perplexity']
            
            # 総合評価スコア計算（正規化）
            all_swuggy = [r['swuggy']['accuracy'] for r in self.results.values()]
            all_sblimp = [r['sblimp']['accuracy'] for r in self.results.values()]
            all_complexity = [r['complexity']['overall_perplexity'] for r in self.results.values()]
            
            norm_swuggy = swuggy_acc / max(all_swuggy) if max(all_swuggy) > 0 else 0
            norm_sblimp = sblimp_acc / max(all_sblimp) if max(all_sblimp) > 0 else 0
            norm_complexity = min(all_complexity) / complexity if complexity > 0 else 0
            
            overall_score = (norm_swuggy + norm_sblimp + norm_complexity) / 3
            
            report += f"| {model_name} | {swuggy_acc:.3f} | {sblimp_acc:.3f} | {complexity:.2f} | {overall_score:.3f} |\n"
        
        # 転移学習効果分析
        comparisons = self.analyze_transfer_learning_effect()
        if comparisons:
            report += "\n## 転移学習効果分析\n\n"
            
            for comp_name, comp_data in comparisons.items():
                model1 = comp_data['models']['model1']
                model2 = comp_data['models']['model2']
                
                report += f"### {comp_name.replace('_', ' ').title()}\n"
                report += f"**{model1} vs {model2}**\n\n"
                
                # 各指標での比較
                for metric in ['swuggy', 'sblimp', 'complexity']:
                    metric_data = comp_data[metric]
                    winner = metric_data['winner']
                    improvement = metric_data['improvement_pct']
                    
                    report += f"- **{metric.title()}**: {winner} (改善度: {improvement:.1f}%)\n"
                
                overall_winner = comp_data['overall_winner']
                report += f"- **総合勝者**: {overall_winner}\n\n"
        
        # 結論
        report += "## 結論\n\n"
        
        # 最優秀モデル特定
        all_swuggy = [(name, results['swuggy']['accuracy']) for name, results in self.results.items()]
        all_sblimp = [(name, results['sblimp']['accuracy']) for name, results in self.results.items()]
        all_complexity = [(name, results['complexity']['overall_perplexity']) for name, results in self.results.items()]
        
        best_swuggy = max(all_swuggy, key=lambda x: x[1])
        best_sblimp = max(all_sblimp, key=lambda x: x[1])
        best_complexity = min(all_complexity, key=lambda x: x[1])
        
        report += f"### 指標別最優秀モデル\n"
        report += f"- **Swuggy**: {best_swuggy[0]} ({best_swuggy[1]:.3f})\n"
        report += f"- **SBLiMP**: {best_sblimp[0]} ({best_sblimp[1]:.3f})\n"
        report += f"- **Complexity**: {best_complexity[0]} ({best_complexity[1]:.2f})\n\n"
        
        # 転移学習効果の総合判定
        if comparisons and 'transfer_vs_scratch' in comparisons:
            transfer_comp = comparisons['transfer_vs_scratch']
            overall_winner = transfer_comp['overall_winner']
            
            if 'transfer' in overall_winner or 'finetuned' in overall_winner:
                report += "### 転移学習効果\n✅ **転移学習の効果が確認されました**\n\n"
                report += "テキスト事前学習により獲得された知識が、音響単位タスクにおいて有効に活用されています。\n\n"
            elif overall_winner == 'tie':
                report += "### 転移学習効果\n⚖️ **混合的な結果**\n\n"
                report += "指標により結果が異なり、転移学習の効果は部分的です。\n\n"
            else:
                report += "### 転移学習効果\n❌ **転移学習の効果は限定的**\n\n"
                report += "スクラッチ学習が転移学習を上回る結果となりました。\n\n"
        
        report += """## 実用的示唆

1. **モデル選択**: 用途に応じた最適なモデル選択の指針
2. **転移学習戦略**: 効果的な転移学習手法の検証
3. **評価手法**: 多角的評価の重要性

## 制限事項

- 限定的なテストデータセット
- 音響単位の簡易的なモデリング
- 評価指標の音響単位タスクへの適応

## 今後の研究方向

- より大規模なデータセットでの検証
- 実際の音声生成品質との相関分析
- 異なる転移学習手法の比較
"""
        
        with open('comprehensive_evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Comprehensive evaluation report generated!")
        return report