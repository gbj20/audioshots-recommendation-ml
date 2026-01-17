import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.inference.recommend import recommend_with_details, get_user_preferences
import pandas as pd
import pickle

# Load data
DATA_PATH = "data/processed/ml_interactions.csv"
AUDIO_META_PATH = "data/processed/audio_language_category.csv"
USER_ENCODER_PATH = "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"

df = pd.read_csv(DATA_PATH)
audio_meta = pd.read_csv(AUDIO_META_PATH)

with open(USER_ENCODER_PATH, "rb") as f:
    user_encoder = pickle.load(f)

with open(ITEM_ENCODER_PATH, "rb") as f:
    item_encoder = pickle.load(f)

# Test with the problematic user
test_user_id = "679b737305a1d9b38bbda765"
user_idx = user_encoder.transform([test_user_id])[0]

print(f"\n{'='*80}")
print(f"TEST USER: {test_user_id}")
print('='*80)

# Show user's preferences
user_data = df[df['user_idx'] == user_idx]
liked_items = user_data[user_data['score'] >= 4.0]

print(f"\nUser Stats:")
print(f"  Total interactions: {len(user_data)}")
print(f"  Liked items (score >= 4): {len(liked_items)}")

# Get preferences
user_prefs = get_user_preferences(user_idx, df, audio_meta, item_encoder)

if user_prefs:
    print(f"\nUser Preferences:")
    print(f"  Preferred Language: {user_prefs['preferred_language']} ({user_prefs['language_weight']*100:.0f}% of likes)")
    print(f"  Preferred Category: {user_prefs['preferred_category']} ({user_prefs['category_weight']*100:.0f}% of likes)")
else:
    print("\n  No preference data available")

# Show user's top liked items
print(f"\nUser's Top Liked Items:")
print("-"*80)

audio_ids = item_encoder.inverse_transform(liked_items['item_idx'].values)
liked_audio_meta = audio_meta[audio_meta['audio_id'].isin(audio_ids)]

for i, (idx, row) in enumerate(liked_audio_meta.head(5).iterrows(), 1):
    print(f"{i}. {row['title'][:50]}")
    print(f"   Language: {row['language']:<15} Category: {row['category']}")

# Get NEW recommendations
print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print('='*80)

recs = recommend_with_details(test_user_id, top_k=10)

if len(recs) > 0:
    # Calculate match rates
    if user_prefs:
        lang_matches = sum(1 for _, row in recs.iterrows() if row['language'] == user_prefs['preferred_language'])
        cat_matches = sum(1 for _, row in recs.iterrows() if row['category'] == user_prefs['preferred_category'])
        
        print(f"\nMatch Rates:")
        print(f"  Language Match: {lang_matches}/10 ({lang_matches*10}%)")
        print(f"  Category Match: {cat_matches}/10 ({cat_matches*10}%)")
    
    print(f"\nTop 5 Recommendations:")
    print("-"*80)
    
    for i, row in recs.head(10).iterrows():
        lang_emoji = "✅" if user_prefs and row['language'] == user_prefs['preferred_language'] else "❌"
        cat_emoji = "✅" if user_prefs and row['category'] == user_prefs['preferred_category'] else "❌"
        
        print(f"{i+1}. [{lang_emoji}] [{cat_emoji}] {row['title'][:45]}")
        print(f"   Audio ID: {row['audio_id']}")
        print(f"   Language: {row['language']:<15} Category: {row['category']:<20}")
        print(f"   Score: {row['predicted_score']:<8} Why: {row['match_reason']}")
        print()
else:
    print("  No recommendations available")

# Test diversity across multiple users
print(f"{'='*80}")
print("DIVERSITY TEST")
print('='*80)

test_users = user_encoder.inverse_transform(range(min(10, len(user_encoder.classes_))))

all_recommendations = []
for test_user in test_users:
    try:
        user_recs = recommend_with_details(test_user, top_k=10)
        all_recommendations.extend(user_recs['audio_id'].tolist())
    except:
        pass

unique_items = len(set(all_recommendations))
total_recs = len(all_recommendations)
diversity = unique_items / total_recs * 100 if total_recs > 0 else 0

print(f"\nResults:")
print(f"  Total recommendations: {total_recs}")
print(f"  Unique items: {unique_items}")
print(f"  Diversity score: {diversity:.1f}%")

# Summary comparison
print(f"\n{'='*80}")
print("BEFORE vs AFTER COMPARISON")
print('='*80)

if user_prefs:
    print(f"   Language Match: {lang_matches*10}%")
    print(f"   Category Match: {cat_matches*10}%")
print(f"   Diversity: {diversity:.0f}%")



