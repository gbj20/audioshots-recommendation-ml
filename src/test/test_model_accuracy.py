"""
TEST MODEL ACCURACY - MANUAL VERIFICATION
This script helps you understand if recommendations make sense
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import pickle
import torch
import torch.nn as nn

# CRITICAL FIX: Import the actual model class instead of redefining it
from src.models.ncf_model import ImprovedNCF

# Paths
MODEL_PATH = "models_saved/recommender_improved.pt"
DATA_PATH = "data/processed/ml_interactions.csv"
ITEM_METADATA_PATH = "data/processed/item_metadata.csv"
AUDIO_META_PATH = "data/processed/audio_language_category.csv"

USER_ENCODER_PATH = "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CRITICAL: Use the same embedding dimension as training (64, not 32)
EMBEDDING_DIM = 64


def test_prediction_accuracy():
    """
    Test 1: Check if model can predict known ratings accurately
    """
    print("\n" + "="*100)
    print("TEST 1: PREDICTION ACCURACY ON KNOWN RATINGS")
    print("="*100 + "\n")
    
    df = pd.read_csv(DATA_PATH)
    
    num_users = df["user_idx"].max() + 1
    num_items = df["item_idx"].max() + 1
    num_languages = df["language_idx"].max() + 1
    num_categories = df["category_idx"].max() + 1
    
    # CRITICAL FIX: Use EMBEDDING_DIM=64 to match training
    model = ImprovedNCF(
        num_users, 
        num_items, 
        num_languages, 
        num_categories, 
        embedding_dim=EMBEDDING_DIM
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)
    
    test_samples = df.sample(min(10, len(df)))
    
    print("Testing model predictions on actual user-item interactions:\n")
    print(f"{'User ID':<15} {'Item ID':<15} {'True Score':<12} {'Predicted':<12} {'Error':<10}")
    print("-"*100)
    
    total_error = 0
    
    for idx, row in test_samples.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        language_idx = int(row['language_idx'])
        category_idx = int(row['category_idx'])
        true_score = float(row['score'])
        
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(DEVICE)
        item_tensor = torch.tensor([item_idx], dtype=torch.long).to(DEVICE)
        language_tensor = torch.tensor([language_idx], dtype=torch.long).to(DEVICE)
        category_tensor = torch.tensor([category_idx], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            prediction = model(user_tensor, item_tensor, language_tensor, category_tensor).item()
        
        error = abs(true_score - prediction)
        total_error += error
        
        user_id = user_encoder.inverse_transform([user_idx])[0]
        item_id = item_encoder.inverse_transform([item_idx])[0]
        
        print(f"{str(user_id)[:15]:<15} {str(item_id)[:15]:<15} {true_score:<12.2f} {prediction:<12.2f} {error:<10.2f}")
    
    avg_error = total_error / len(test_samples)
    
    print("-"*100)
    print(f"\nAverage Error: {avg_error:.4f}")
    
    if avg_error < 0.5:
        print("✓ EXCELLENT - Model is very accurate on known interactions")
    elif avg_error < 1.0:
        print("✓ GOOD - Model has reasonable accuracy")
    elif avg_error < 1.5:
        print("⚠ MODERATE - Model is okay but could be better")
    else:
        print("✗ POOR - Model needs improvement")
    
    print()


def test_recommendation_relevance():
    """
    Test 2: Check if recommended items are relevant to user's history
    """
    print("\n" + "="*100)
    print("TEST 2: RECOMMENDATION RELEVANCE")
    print("="*100 + "\n")
    
    df = pd.read_csv(DATA_PATH)
    audio_meta = pd.read_csv(AUDIO_META_PATH)
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)
    
    user_counts = df.groupby('user_idx').size()
    active_user_idx = user_counts.idxmax()
    
    user_id = user_encoder.inverse_transform([active_user_idx])[0]
    
    print(f"Testing User: {user_id}")
    print(f"This user has {user_counts[active_user_idx]} interactions\n")
    
    user_interactions = df[df['user_idx'] == active_user_idx].sort_values('score', ascending=False)
    
    print("USER'S TOP LIKED ITEMS:")
    print("-"*100)
    
    for i, (idx, row) in enumerate(user_interactions.head(5).iterrows(), 1):
        item_id = item_encoder.inverse_transform([int(row['item_idx'])])[0]
        audio_info = audio_meta[audio_meta['audio_id'] == item_id]
        
        if len(audio_info) > 0:
            audio_info = audio_info.iloc[0]
            print(f"{i}. {audio_info['title'][:50]:<50}")
            print(f"   Language: {audio_info['language']:<15} Category: {audio_info['category']:<20} Score: {row['score']:.1f}")
        print()
    
    from src.inference.recommend import recommend_with_details
    
    print("\nRECOMMENDATIONS FOR THIS USER:")
    print("-"*100)
    
    recommendations = recommend_with_details(user_id, top_k=5)
    
    if len(recommendations) > 0:
        for i, row in recommendations.iterrows():
            print(f"{i+1}. {row['title'][:50]:<50}")
            print(f"   Language: {row['language']:<15} Category: {row['category']:<20} Predicted: {row['predicted_score']}")
            print()
        
        user_languages = user_interactions.merge(
            audio_meta[['audio_id', 'language', 'category']], 
            left_on=item_encoder.inverse_transform(user_interactions['item_idx']),
            right_on='audio_id',
            how='left'
        )
        
        most_common_language = user_languages['language'].mode()[0] if len(user_languages) > 0 else "Unknown"
        most_common_category = user_languages['category'].mode()[0] if len(user_languages) > 0 else "Unknown"
        
        rec_languages = recommendations['language'].value_counts()
        rec_categories = recommendations['category'].value_counts()
        
        print("\nRELEVANCE CHECK:")
        print(f"  User's most common language: {most_common_language}")
        print(f"  Recommendations matching language: {rec_languages.get(most_common_language, 0)}/{len(recommendations)}")
        print(f"  User's most common category: {most_common_category}")
        print(f"  Recommendations matching category: {rec_categories.get(most_common_category, 0)}/{len(recommendations)}")
        
        language_match_rate = rec_languages.get(most_common_language, 0) / len(recommendations)
        
        if language_match_rate > 0.6:
            print("\n  ✓ GOOD - Recommendations align well with user preferences")
        elif language_match_rate > 0.3:
            print("\n  ⚠ MODERATE - Some alignment with user preferences")
        else:
            print("\n  ✗ POOR - Recommendations don't match user preferences")
    else:
        print("  No recommendations available")
    
    print()


def test_diversity():
    """
    Test 3: Check if model recommends diverse items
    """
    print("\n" + "="*100)
    print("TEST 3: RECOMMENDATION DIVERSITY")
    print("="*100 + "\n")
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    from src.inference.recommend import recommend_for_user
    
    num_test_users = min(10, len(user_encoder.classes_))
    
    all_recommendations = []
    
    print(f"Testing recommendations for {num_test_users} different users...\n")
    
    for i in range(num_test_users):
        user_id = user_encoder.inverse_transform([i])[0]
        recs = recommend_for_user(user_id, top_k=10)
        all_recommendations.extend(recs)
    
    unique_items = len(set(all_recommendations))
    total_recommendations = len(all_recommendations)
    diversity_score = unique_items / total_recommendations if total_recommendations > 0 else 0
    
    print(f"Total recommendations made: {total_recommendations}")
    print(f"Unique items recommended: {unique_items}")
    print(f"Diversity score: {diversity_score:.2%}")
    print()
    
    if diversity_score > 0.7:
        print("✓ EXCELLENT - Model recommends diverse items to different users")
    elif diversity_score > 0.4:
        print("✓ GOOD - Model has reasonable diversity")
    elif diversity_score > 0.2:
        print("⚠ MODERATE - Model could be more diverse")
    else:
        print("✗ POOR - Model recommends same items to everyone (popularity bias)")
    
    print()


def test_cold_start():
    """
    Test 4: Check how model handles unknown users
    """
    print("\n" + "="*100)
    print("TEST 4: COLD START HANDLING")
    print("="*100 + "\n")
    
    from src.inference.recommend import recommend_for_user
    
    fake_user_id = "UNKNOWN_USER_12345"
    
    print(f"Testing with unknown user: {fake_user_id}")
    print("Model should return popular items...\n")
    
    recs = recommend_for_user(fake_user_id, top_k=5)
    
    if len(recs) > 0:
        print("✓ GOOD - Model handles cold start gracefully")
        print(f"  Returned {len(recs)} popular items")
    else:
        print("✗ POOR - Model failed to handle unknown user")
    
    print()


def overall_summary():
    """
    Overall summary of model quality
    """
    print("\n" + "="*100)
    print("OVERALL MODEL QUALITY SUMMARY")
    print("="*100 + "\n")
    
    df = pd.read_csv(DATA_PATH)
    
    print("MODEL STATISTICS:")
    print(f"  Training data size: {len(df):,} interactions")
    print(f"  Number of users: {df['user_idx'].nunique():,}")
    print(f"  Number of items: {df['item_idx'].nunique():,}")
    print(f"  Data sparsity: {1 - len(df) / (df['user_idx'].nunique() * df['item_idx'].nunique()):.4%}")
    print()
    
    print("QUALITY CHECKLIST:")
    print("  ☐ MAE < 1.0 (Check training output)")
    print("  ☐ Predictions match known ratings well (Test 1)")
    print("  ☐ Recommendations align with user preferences (Test 2)")
    print("  ☐ Diversity score > 40% (Test 3)")
    print("  ☐ Cold start handled properly (Test 4)")
    print()
    
    print("IF ALL TESTS PASS:")
    print("  ✓ Your model is production-ready!")
    print()
    print("IF SOME TESTS FAIL:")
    print("  • Low accuracy → Need more data or better features")
    print("  • Poor relevance → Check if language/category features are working")
    print("  • Low diversity → May need to add exploration/randomness")
    print("  • Cold start issues → Implement better fallback strategy")
    print()


if __name__ == "__main__":
    print("\n" + "="*100)
    print("MODEL ACCURACY TESTING SUITE")
    print("="*100)
    
    test_prediction_accuracy()
    test_recommendation_relevance()
    test_diversity()
    test_cold_start()
    overall_summary()
    
    print("="*100)
    print("TESTING COMPLETE!")
    print("="*100 + "\n")