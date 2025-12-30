import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
import pickle
from collections import defaultdict

from src.models.ncf_model import NCF

# ---------------- CONFIG ----------------
MODEL_PATH = "models_saved/recommender.pt"
DATA_PATH = "data/processed/ml_interactions.csv"
AUDIO_DATA_PATH = "data/processed/audio_language_category.csv"

USER_ENCODER_PATH = "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"
LANGUAGE_ENCODER_PATH = "models_saved/language_encoder.pkl"
CATEGORY_ENCODER_PATH = "models_saved/category_encoder.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HybridRecommender:
    """Hybrid recommender with content filtering"""
    
    def __init__(self):
        # Load model and encoders
        df = pd.read_csv(DATA_PATH)
        
        with open(USER_ENCODER_PATH, "rb") as f:
            self.user_encoder = pickle.load(f)
        with open(ITEM_ENCODER_PATH, "rb") as f:
            self.item_encoder = pickle.load(f)
        with open(LANGUAGE_ENCODER_PATH, "rb") as f:
            self.language_encoder = pickle.load(f)
        with open(CATEGORY_ENCODER_PATH, "rb") as f:
            self.category_encoder = pickle.load(f)
        
        num_users = df["user_idx"].nunique()
        num_items = df["item_idx"].nunique()
        num_languages = df["language_idx"].nunique()
        num_categories = df["category_idx"].nunique()
        
        self.model = NCF(num_users, num_items, num_languages, num_categories)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        
        # Load audio metadata
        audio_df = pd.read_csv(AUDIO_DATA_PATH)
        
        # Build item metadata lookup
        # IMPORTANT: Don't group by audio_id because one audio can have multiple categories
        # Instead, create multiple entries for each audio-category pair
        self.item_metadata = {}
        self.item_to_audio = {}  # Map item_idx -> audio_id
        
        for _, row in audio_df.iterrows():
            audio_id = row['audio_id']
            language = row['language']
            category = row['category']
            
            if pd.notna(audio_id) and pd.notna(language) and pd.notna(category):
                if audio_id in self.item_encoder.classes_:
                    item_idx = self.item_encoder.transform([audio_id])[0]
                    
                    # Create a unique key for each item-language-category combo
                    key = (item_idx, language, category)
                    
                    self.item_metadata[key] = {
                        'audio_id': audio_id,
                        'language': language,
                        'category': category
                    }
                    
                    self.item_to_audio[item_idx] = audio_id
    
    def recommend(self, user_id, language, category, top_k=10):
        """Recommend items filtered by language and category"""
        
        # Check if language and category are known to the encoders
        try:
            language_idx = self.language_encoder.transform([language])[0]
        except ValueError:
            # Language not in training data
            return []
        
        try:
            category_idx = self.category_encoder.transform([category])[0]
        except ValueError:
            # Category not in training data
            return []
        
        # Step 1: Filter items by language and category
        # Look through all item-language-category combinations
        filtered_item_indices = set()
        
        for key, metadata in self.item_metadata.items():
            item_idx, meta_lang, meta_cat = key
            if meta_lang == language and meta_cat == category:
                filtered_item_indices.add(item_idx)
        
        filtered_item_indices = list(filtered_item_indices)
        
        if len(filtered_item_indices) == 0:
            return []
        
        # Step 2: Get user index
        if user_id not in self.user_encoder.classes_:
            user_id = self.user_encoder.classes_[0]
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Step 4: Score the FILTERED items
        item_tensor = torch.tensor(filtered_item_indices, dtype=torch.long).to(DEVICE)
        user_tensor = torch.full((len(filtered_item_indices),), user_idx, dtype=torch.long).to(DEVICE)
        language_tensor = torch.full((len(filtered_item_indices),), language_idx, dtype=torch.long).to(DEVICE)
        category_tensor = torch.full((len(filtered_item_indices),), category_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor, language_tensor, category_tensor)
        
        # Step 5: Get top-K
        # Handle case where scores is 0-d (single item)
        if scores.dim() == 0:
            # Only one item, return it
            recommended_audio_ids = self.item_encoder.inverse_transform([filtered_item_indices[0]])
        else:
            k = min(top_k, len(scores))
            if k == 1:
                top_indices = [torch.argmax(scores).item()]
            else:
                top_indices = torch.topk(scores, k).indices.cpu().numpy()
            top_item_indices = [filtered_item_indices[i] for i in top_indices]
            recommended_audio_ids = self.item_encoder.inverse_transform(top_item_indices)
        
        return list(recommended_audio_ids)


# ---------------- VALIDATION ----------------
def validate():
    print("\n" + "="*70)
    print("LOADING HYBRID RECOMMENDER FOR VALIDATION")
    print("="*70 + "\n")
    
    recommender = HybridRecommender()
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Device: {DEVICE}")
    print(f"✓ Item metadata loaded: {len(recommender.item_metadata)} items\n")
    
    # Build ground truth from audio metadata
    audio_df = pd.read_csv(AUDIO_DATA_PATH)
    ground_truth = defaultdict(lambda: defaultdict(set))
    
    for _, row in audio_df.iterrows():
        audio_id = row['audio_id']
        language = row['language']
        category = row['category']
        
        if pd.notna(audio_id) and pd.notna(language) and pd.notna(category):
            ground_truth[language][category].add(audio_id)
    
    print(f"✓ Built ground truth for {len(ground_truth)} languages\n")
    
    # Get available combinations
    available_combinations = []
    for language in sorted(ground_truth.keys()):
        for category in sorted(ground_truth[language].keys()):
            num_audios = len(ground_truth[language][category])
            if num_audios > 0:
                available_combinations.append((language, category, num_audios))
    
    print(f"✓ Found {len(available_combinations)} language-category combinations\n")
    
    # Interactive validation
    while True:
        print("="*70)
        print("AVAILABLE LANGUAGES")
        print("="*70 + "\n")
        
        languages = sorted(ground_truth.keys())
        for idx, lang in enumerate(languages, 1):
            num_cats = len(ground_truth[lang])
            total_audios = sum(len(ground_truth[lang][cat]) for cat in ground_truth[lang])
            print(f"{idx:2d}. {lang:20s} → {num_cats} categories, {total_audios} audio IDs")
        
        print()
        lang_input = input("Enter language name (or 'exit' to quit): ").strip()
        
        if lang_input.lower() == 'exit':
            print("\n✓ Validation complete!\n")
            break
        
        if lang_input not in languages:
            print(f"\n✗ Language '{lang_input}' not found!\n")
            continue
        
        # Show categories
        print(f"\n{'='*70}")
        print(f"CATEGORIES FOR: {lang_input}")
        print(f"{'='*70}\n")
        
        categories = sorted(ground_truth[lang_input].keys())
        for idx, cat in enumerate(categories, 1):
            num_audios = len(ground_truth[lang_input][cat])
            print(f"{idx:2d}. {cat:40s} → {num_audios} audio IDs")
        
        print()
        cat_input = input("Enter category name (or 'back'): ").strip()
        
        if cat_input.lower() == 'back':
            continue
        
        if cat_input not in categories:
            print(f"\n✗ Category '{cat_input}' not found!\n")
            continue
        
        # Get recommendations
        top_k = 10
        user_id = recommender.user_encoder.classes_[0]
        
        print(f"\n{'='*70}")
        print(f"GETTING RECOMMENDATIONS: {lang_input} → {cat_input}")
        print(f"{'='*70}\n")
        
        recommendations = recommender.recommend(user_id, lang_input, cat_input, top_k)
        
        # Validate
        true_audio_ids = ground_truth[lang_input][cat_input]
        correct = set(recommendations).intersection(true_audio_ids)
        
        precision = len(correct) / len(recommendations) if recommendations else 0
        recall = len(correct) / len(true_audio_ids) if true_audio_ids else 0
        
        print(f"{'='*70}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*70}\n")
        
        print(f"📊 Ground Truth Audio IDs    : {len(true_audio_ids)}")
        print(f"🎯 Recommended Audio IDs     : {len(recommendations)}")
        print(f"✓  Correct Recommendations   : {len(correct)}")
        print(f"\n📈 Precision                 : {precision:.2%}")
        print(f"📈 Recall                    : {recall:.2%}")
        
        # Interpretation
        print(f"\n{'─'*70}")
        print("📝 RESULT:")
        print(f"{'─'*70}")
        
        if precision == 1.0:
            print("✅ PERFECT! All recommendations match the language-category filter!")
        elif precision >= 0.9:
            print("✅ EXCELLENT! Nearly all recommendations are correct!")
        elif precision >= 0.7:
            print("✓ GOOD! Most recommendations are correct.")
        else:
            print("⚠ Some recommendations don't match the filter.")
        
        print()
        
        # Show sample recommendations
        show_sample = input("Show sample recommendations? (y/n): ").strip().lower()
        if show_sample == 'y':
            print(f"\n{'─'*70}")
            print("RECOMMENDED AUDIO IDs:")
            print(f"{'─'*70}")
            for i, aid in enumerate(recommendations[:10], 1):
                status = "✓" if aid in true_audio_ids else "✗"
                print(f"  {status} {i:2d}. {aid}")
            print()
        
        another = input("Test another combination? (y/n): ").strip().lower()
        if another != 'y':
            print("\n✓ Validation complete!\n")
            print("="*70 + "\n")
            break


# ---------------- BATCH VALIDATION ----------------
def batch_validate():
    print("\n" + "="*70)
    print("BATCH VALIDATION - ALL COMBINATIONS")
    print("="*70 + "\n")
    
    recommender = HybridRecommender()
    
    # Build ground truth
    audio_df = pd.read_csv(AUDIO_DATA_PATH)
    ground_truth = defaultdict(lambda: defaultdict(set))
    
    for _, row in audio_df.iterrows():
        audio_id = row['audio_id']
        language = row['language']
        category = row['category']
        
        if pd.notna(audio_id) and pd.notna(language) and pd.notna(category):
            ground_truth[language][category].add(audio_id)
    
    # Get user
    user_id = recommender.user_encoder.classes_[0]
    
    # Test all combinations
    results = []
    skipped = []
    total = sum(len(ground_truth[lang]) for lang in ground_truth)
    count = 0
    
    for language in sorted(ground_truth.keys()):
        for category in sorted(ground_truth[language].keys()):
            count += 1
            print(f"\r[{count}/{total}] Testing {language} → {category}...", end='')
            
            recommendations = recommender.recommend(user_id, language, category, top_k=10)
            true_audio_ids = ground_truth[language][category]
            
            # Skip if category not in training data (returns empty list)
            if len(recommendations) == 0 and len(true_audio_ids) > 0:
                # Check if it's because category is unknown
                try:
                    recommender.category_encoder.transform([category])
                    # Category is known, but no recommendations - could be other issue
                    correct = set()
                    precision = 0.0
                except ValueError:
                    # Category not in training data - skip
                    skipped.append((language, category, len(true_audio_ids)))
                    continue
            
            correct = set(recommendations).intersection(true_audio_ids)
            
            precision = len(correct) / len(recommendations) if recommendations else 0
            recall = len(correct) / len(true_audio_ids) if true_audio_ids else 0
            
            results.append({
                'language': language,
                'category': category,
                'precision': precision,
                'recall': recall,
                'ground_truth': len(true_audio_ids),
                'recommended': len(recommendations),
                'correct': len(correct)
            })
    
    print("\n")
    
    # Summary
    avg_precision = sum(r['precision'] for r in results) / len(results) if results else 0
    avg_recall = sum(r['recall'] for r in results) / len(results) if results else 0
    perfect_count = sum(1 for r in results if r['precision'] == 1.0)
    
    print(f"\n{'='*70}")
    print("BATCH VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Total combinations tested : {len(results)}")
    if skipped:
        print(f"Skipped (not in training) : {len(skipped)}")
        print(f"  (Categories in metadata but never seen during training)")
    print(f"Perfect precision (100%)  : {perfect_count} ({perfect_count/len(results)*100:.1f}%)")
    print(f"Average Precision         : {avg_precision:.2%}")
    print(f"Average Recall            : {avg_recall:.2%}")
    
    print(f"\n{'─'*70}")
    print("OVERALL RESULT:")
    print(f"{'─'*70}")
    
    if avg_precision >= 0.95:
        print("✅ EXCELLENT! The hybrid recommender works perfectly!")
    elif avg_precision >= 0.80:
        print("✓ GOOD! The recommender filters correctly in most cases.")
    else:
        print("⚠ There are some issues with filtering accuracy.")
    
    # Show worst performers
    results_sorted = sorted(results, key=lambda x: x['precision'])
    
    print(f"\n{'─'*70}")
    print("LOWEST PRECISION (if any):")
    print(f"{'─'*70}")
    for r in results_sorted[:5]:
        if r['precision'] < 1.0:
            print(f"{r['language']:15s} → {r['category']:25s} | P: {r['precision']:.2%}, Correct: {r['correct']}/{r['recommended']}")
    
    print("\n" + "="*70 + "\n")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\nChoose validation mode:")
    print("1. Interactive validation (test specific combinations)")
    print("2. Batch validation (test all combinations)")
    
    mode = input("\nEnter choice (1-2): ").strip()
    
    if mode == '2':
        batch_validate()
    else:
        validate()