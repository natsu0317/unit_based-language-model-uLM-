# === run_comprehensive_evaluation.py ===
from comprehensive_evaluator import ComprehensiveEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch

def main():
    print("=== COMPREHENSIVE TRANSFER LEARNING EVALUATION ===")
    print("Evaluating with Swuggy, SBLiMP, and Complexity metrics")
    
    # 1. テストデータ読み込み
    print("\n📚 Loading test data...")
    test_data = pd.read_csv('../units_dev.csv')
    print(f"Test data: {len(test_data)} samples")
    
    # 2. トークナイザー準備
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. モデル準備
    print("\n🤖 Loading models...")
    
    models = {
        'qwen_transfer': {
            'model': AutoModelForCausalLM.from_pretrained(
                "../qwen-unit-lm/ulm-qwen3b-merged",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ),
            'tokenizer': tokenizer,
            'type': 'transfer_learning',
            'description': 'Qwen pretrained + acoustic unit finetuning'
        },
        
        'gpt2_scratch': {
            'model': AutoModelForCausalLM.from_pretrained(
                "../gpt2-unit-lm/ulm-gpt2-scratch-final",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ),
            'tokenizer': tokenizer,
            'type': 'scratch_learning',
            'description': 'GPT2 trained from scratch on acoustic units'
        },
        
        # オプション: ベースラインモデル
        # 'qwen_baseline': {
        #     'model': AutoModelForCausalLM.from_pretrained(
        #         "Qwen/Qwen2.5-3B",
        #         torch_dtype=torch.bfloat16,
        #         device_map="auto"
        #     ),
        #     'tokenizer': tokenizer,
        #     'type': 'pretrained_baseline',
        #     'description': 'Original Qwen pretrained (no acoustic unit training)'
        # }
    }
    
    print(f"Models loaded: {list(models.keys())}")
    
    # 4. 包括的評価実行
    print("\n🔬 Starting comprehensive evaluation...")
    evaluator = ComprehensiveEvaluator()
    
    # 全モデル評価
    all_results = evaluator.evaluate_all_models(models, test_data)
    
    # 転移学習効果分析
    print("\n📊 Analyzing transfer learning effects...")
    comparisons = evaluator.analyze_transfer_learning_effect()
    
    # 可視化作成
    print("\n📈 Creating visualizations...")
    evaluator.create_comprehensive_visualization()
    
    # レポート生成
    print("\n📝 Generating comprehensive report...")
    evaluator.generate_comprehensive_report()
    
    # 結果サマリー表示
    print("\n" + "="*60)
    print("🎉 EVALUATION COMPLETED!")
    print("="*60)
    
    print("\n📊 RESULTS SUMMARY:")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Swuggy: {results['swuggy']['accuracy']:.3f}")
        print(f"  SBLiMP: {results['sblimp']['accuracy']:.3f}")
        print(f"  Complexity: {results['complexity']['overall_perplexity']:.2f}")
    
    if comparisons and 'transfer_vs_scratch' in comparisons:
        comp = comparisons['transfer_vs_scratch']
        print(f"\n🏆 TRANSFER LEARNING EFFECT:")
        print(f"  Overall Winner: {comp['overall_winner']}")
        
        if 'transfer' in comp['overall_winner'] or 'finetuned' in comp['overall_winner']:
            print("  ✅ Transfer learning is effective!")
        else:
            print("  ⚠️ Transfer learning effect is limited.")
    
    print(f"\n📁 Output files:")
    print(f"  - comprehensive_evaluation_results.png")
    print(f"  - comprehensive_evaluation_report.md")

if __name__ == "__main__":
    main()