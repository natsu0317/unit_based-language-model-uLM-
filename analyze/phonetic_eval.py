import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

class PhoneticTransferAnalyzer:
    def __init__(self, model1_path, model2_path):
        """
        model1_path: "qwen-3B-final"
        model2_path: "../gpt2-unit-lm/ulm-gpt2-scratch-final"
        """
        print("🤖 Loading models...")
        self.tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
        self.model1 = AutoModel.from_pretrained(model1_path)
        
        self.tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
        self.model2 = AutoModel.from_pretrained(model2_path)
        
        self.model1.eval()
        self.model2.eval()
        
    def get_word_embedding(self, word, model, tokenizer, layer=-2):
        """単語の内部表現を抽出"""
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 指定層の平均プーリング
            hidden_state = outputs.hidden_states[layer]
            embedding = hidden_state.mean(dim=1).squeeze()
        return embedding.numpy()
    
    def create_phonetic_dataset(self):
        """音韻実験用データセット作成"""
        # 音韻的類似性の異なる単語ペア
        phonetic_pairs = [
            # 1音素差（最小対立ペア）
            ("cat", "bat", 1), ("cat", "rat", 1), ("cat", "hat", 1),
            ("pen", "ten", 1), ("pen", "hen", 1), ("big", "bag", 1),
            ("sit", "set", 1), ("sit", "bit", 1), ("run", "sun", 1),
            
            # 2音素差
            ("cat", "cut", 2), ("pen", "pin", 2), ("dog", "dig", 2),
            ("car", "bar", 2), ("red", "bed", 2), ("hot", "hat", 2),
            
            # 3音素差以上
            ("cat", "dog", 3), ("pen", "car", 3), ("big", "sun", 3),
            ("run", "hat", 3), ("red", "big", 3), ("hot", "pen", 4),
            
            # 音韻的に全く異なる
            ("elephant", "quick", 5), ("beautiful", "dog", 5),
            ("computer", "sun", 5), ("language", "cat", 5),
        ]
        
        return phonetic_pairs
    
    def create_phonetic_features_dataset(self):
        """音韻特徴分類用データセット"""
        vowels = ["a", "e", "i", "o", "u", "apple", "elephant", "igloo", "orange", "umbrella"]
        consonants = ["b", "c", "d", "f", "g", "book", "cat", "dog", "fish", "game"]
        
        voiced = ["b", "d", "g", "v", "z", "book", "dog", "game", "voice", "zero"]
        voiceless = ["p", "t", "k", "f", "s", "pen", "top", "cat", "fish", "sun"]
        
        return {
            'vowel_consonant': (vowels, consonants),
            'voiced_voiceless': (voiced, voiceless)
        }
    
    def analyze_phonetic_similarity_preservation(self):
        """音韻類似性保存の分析"""
        print("📊 Analyzing phonetic similarity preservation...")
        
        pairs = self.create_phonetic_dataset()
        results = {'qwen': [], 'gpt2': [], 'phonetic_dist': []}
        
        for word1, word2, phon_dist in pairs:
            try:
                # Qwen embeddings
                emb1_qwen = self.get_word_embedding(word1, self.model1, self.tokenizer1)
                emb2_qwen = self.get_word_embedding(word2, self.model1, self.tokenizer1)
                cos_dist_qwen = cosine(emb1_qwen, emb2_qwen)
                
                # GPT-2 embeddings  
                emb1_gpt2 = self.get_word_embedding(word1, self.model2, self.tokenizer2)
                emb2_gpt2 = self.get_word_embedding(word2, self.model2, self.tokenizer2)
                cos_dist_gpt2 = cosine(emb1_gpt2, emb2_gpt2)
                
                results['qwen'].append(cos_dist_qwen)
                results['gpt2'].append(cos_dist_gpt2)
                results['phonetic_dist'].append(phon_dist)
                
                print(f"  {word1}-{word2}: Qwen={cos_dist_qwen:.3f}, GPT2={cos_dist_gpt2:.3f}, Phon={phon_dist}")
                
            except Exception as e:
                print(f"  Error with {word1}-{word2}: {e}")
                continue
        
        # 相関分析
        corr_qwen = pearsonr(results['phonetic_dist'], results['qwen'])
        corr_gpt2 = pearsonr(results['phonetic_dist'], results['gpt2'])
        
        print(f"\n📈 Correlation Results:")
        print(f"  Qwen: r={corr_qwen[0]:.4f}, p={corr_qwen[1]:.4f}")
        print(f"  GPT2: r={corr_gpt2[0]:.4f}, p={corr_gpt2[1]:.4f}")
        
        # 可視化
        self.plot_correlation(results, corr_qwen, corr_gpt2)
        
        return results, corr_qwen, corr_gpt2
    
    def analyze_phonetic_feature_classification(self):
        """音韻特徴分類分析"""
        print("🔍 Analyzing phonetic feature classification...")
        
        features_data = self.create_phonetic_features_dataset()
        results = {}
        
        for feature_name, (class1_words, class2_words) in features_data.items():
            print(f"\n  Testing {feature_name} classification...")
            
            # データ準備
            X_qwen, X_gpt2, y = [], [], []
            
            for word in class1_words:
                try:
                    emb_qwen = self.get_word_embedding(word, self.model1, self.tokenizer1)
                    emb_gpt2 = self.get_word_embedding(word, self.model2, self.tokenizer2)
                    X_qwen.append(emb_qwen)
                    X_gpt2.append(emb_gpt2)
                    y.append(0)
                except:
                    continue
                    
            for word in class2_words:
                try:
                    emb_qwen = self.get_word_embedding(word, self.model1, self.tokenizer1)
                    emb_gpt2 = self.get_word_embedding(word, self.model2, self.tokenizer2)
                    X_qwen.append(emb_qwen)
                    X_gpt2.append(emb_gpt2)
                    y.append(1)
                except:
                    continue
            
            X_qwen = np.array(X_qwen)
            X_gpt2 = np.array(X_gpt2)
            y = np.array(y)
            
            # 分類器訓練・評価
            clf_qwen = LogisticRegression(random_state=42)
            clf_gpt2 = LogisticRegression(random_state=42)
            
            clf_qwen.fit(X_qwen, y)
            clf_gpt2.fit(X_gpt2, y)
            
            acc_qwen = clf_qwen.score(X_qwen, y)
            acc_gpt2 = clf_gpt2.score(X_gpt2, y)
            
            results[feature_name] = {
                'qwen_accuracy': acc_qwen,
                'gpt2_accuracy': acc_gpt2
            }
            
            print(f"    Qwen accuracy: {acc_qwen:.3f}")
            print(f"    GPT2 accuracy: {acc_gpt2:.3f}")
        
        return results
    
    def analyze_layer_wise_transfer(self):
        """層別転移効果分析"""
        print("🔬 Analyzing layer-wise transfer effects...")
        
        test_pairs = [("cat", "bat", 1), ("pen", "ten", 1), ("dog", "dig", 2)]
        layers_to_test = [-1, -2, -3, -4, -5]  # 最後の5層
        
        results = {'layer': [], 'qwen_corr': [], 'gpt2_corr': []}
        
        for layer in layers_to_test:
            print(f"  Testing layer {layer}...")
            
            qwen_dists, gpt2_dists, phon_dists = [], [], []
            
            for word1, word2, phon_dist in test_pairs:
                try:
                    # 指定層での埋め込み取得
                    emb1_qwen = self.get_word_embedding(word1, self.model1, self.tokenizer1, layer)
                    emb2_qwen = self.get_word_embedding(word2, self.model1, self.tokenizer1, layer)
                    
                    emb1_gpt2 = self.get_word_embedding(word1, self.model2, self.tokenizer2, layer)
                    emb2_gpt2 = self.get_word_embedding(word2, self.model2, self.tokenizer2, layer)
                    
                    qwen_dists.append(cosine(emb1_qwen, emb2_qwen))
                    gpt2_dists.append(cosine(emb1_gpt2, emb2_gpt2))
                    phon_dists.append(phon_dist)
                except:
                    continue
            
            if len(qwen_dists) > 2:
                corr_qwen = pearsonr(phon_dists, qwen_dists)[0]
                corr_gpt2 = pearsonr(phon_dists, gpt2_dists)[0]
                
                results['layer'].append(layer)
                results['qwen_corr'].append(corr_qwen)
                results['gpt2_corr'].append(corr_gpt2)
                
                print(f"    Layer {layer}: Qwen r={corr_qwen:.3f}, GPT2 r={corr_gpt2:.3f}")
        
        return results
    
    def plot_correlation(self, results, corr_qwen, corr_gpt2):
        """相関結果の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Qwen
        ax1.scatter(results['phonetic_dist'], results['qwen'], alpha=0.7)
        ax1.set_xlabel('Phonetic Distance')
        ax1.set_ylabel('Embedding Distance (Cosine)')
        ax1.set_title(f'Qwen: r={corr_qwen[0]:.3f}, p={corr_qwen[1]:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # GPT-2
        ax2.scatter(results['phonetic_dist'], results['gpt2'], alpha=0.7, color='orange')
        ax2.set_xlabel('Phonetic Distance')
        ax2.set_ylabel('Embedding Distance (Cosine)')
        ax2.set_title(f'GPT-2: r={corr_gpt2[0]:.3f}, p={corr_gpt2[1]:.3f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('phonetic_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """完全な分析実行"""
        print("🚀 Starting Phonetic Transfer Analysis")
        print("="*50)
        
        # 1. 音韻類似性保存分析
        similarity_results, corr_qwen, corr_gpt2 = self.analyze_phonetic_similarity_preservation()
        
        # 2. 音韻特徴分類分析
        classification_results = self.analyze_phonetic_feature_classification()
        
        # 3. 層別分析
        layer_results = self.analyze_layer_wise_transfer()
        
        # 4. 総合結果
        print("\n" + "="*50)
        print("📋 FINAL RESULTS")
        print("="*50)
        
        print(f"🎯 Phonetic Similarity Preservation:")
        print(f"   Qwen correlation: r={corr_qwen[0]:.4f} (p={corr_qwen[1]:.4f})")
        print(f"   GPT2 correlation: r={corr_gpt2[0]:.4f} (p={corr_gpt2[1]:.4f})")
        
        if corr_qwen[0] > corr_gpt2[0] and corr_qwen[1] < 0.05:
            print("   ✅ Qwen shows superior phonetic structure preservation!")
        
        print(f"\n🔍 Phonetic Feature Classification:")
        for feature, acc in classification_results.items():
            print(f"   {feature}:")
            print(f"     Qwen: {acc['qwen_accuracy']:.3f}")
            print(f"     GPT2: {acc['gpt2_accuracy']:.3f}")
            if acc['qwen_accuracy'] > acc['gpt2_accuracy']:
                print(f"     ✅ Qwen superior")
        
        # 証明判定
        qwen_superior = (
            corr_qwen[0] > corr_gpt2[0] and 
            corr_qwen[1] < 0.05 and
            corr_qwen[0] > 0.3
        )
        
        print(f"\n🏆 CONCLUSION:")
        if qwen_superior:
            print("   ✅ PROVEN: Qwen's text representations contain phonetic structure")
            print("   ✅ This explains superior transfer to phoneme learning!")
        else:
            print("   ❌ No clear evidence of phonetic structure preservation")
        
        return {
            'similarity': (corr_qwen, corr_gpt2),
            'classification': classification_results,
            'layers': layer_results,
            'proven': qwen_superior
        }

# 実行コード
if __name__ == "__main__":
    analyzer = PhoneticTransferAnalyzer(
        model1_path="qwen-3B-final",
        model2_path="../gpt2-unit-lm/ulm-gpt2-scratch-final"
    )
    
    results = analyzer.run_complete_analysis()