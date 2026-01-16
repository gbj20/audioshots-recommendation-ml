# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(PROJECT_ROOT))

# from src.models.ncf_model import ImprovedNCF
# import pickle
# import pandas as pd
# import torch
# import numpy as np

# # Paths
# MODEL_PATH = PROJECT_ROOT / "models_saved/recommender_improved.pt"
# DATA_PATH = PROJECT_ROOT / "data/processed/ml_interactions.csv"
# ITEM_METADATA_PATH = PROJECT_ROOT / "data/processed/item_metadata.csv"

# USER_ENCODER_PATH = PROJECT_ROOT / "models_saved/user_encoder.pkl"
# ITEM_ENCODER_PATH = PROJECT_ROOT / "models_saved/item_encoder.pkl"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# def load_model_and_encoders():
#     """Load trained model and encoders"""
    
#     # Load data to get dimensions
#     df = pd.read_csv(DATA_PATH)
    
#     num_users = df["user_idx"].max() + 1
#     num_items = df["item_idx"].max() + 1
#     num_languages = df["language_idx"].max() + 1
#     num_categories = df["category_idx"].max() + 1
    
#     # Load model
#     model = ImprovedNCF(
#         num_users=num_users,
#         num_items=num_items,
#         num_languages=num_languages,
#         num_categories=num_categories,
#         embedding_dim=64
#     )
    
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
    
#     # Load encoders
#     with open(USER_ENCODER_PATH, "rb") as f:
#         user_encoder = pickle.load(f)
    
#     with open(ITEM_ENCODER_PATH, "rb") as f:
#         item_encoder = pickle.load(f)
    
#     # Load item metadata
#     item_metadata = pd.read_csv(ITEM_METADATA_PATH)
    
#     return model, user_encoder, item_encoder, item_metadata, df


# def get_user_preferences(user_idx, df, audio_meta, item_encoder):
#     """
#     NEW FUNCTION: Extract user's language and category preferences
    
#     Returns:
#         dict with preferred_language, preferred_category
#     """
#     user_data = df[df['user_idx'] == user_idx]
    
#     if len(user_data) == 0:
#         return None
    
#     # Get items user liked (score >= 4)
#     liked_items = user_data[user_data['score'] >= 4.0]
    
#     # If no strong likes, use all interactions
#     if len(liked_items) < 2:
#         liked_items = user_data
    
#     # Get audio metadata for liked items
#     audio_ids = item_encoder.inverse_transform(liked_items['item_idx'].values)
#     items_meta = audio_meta[audio_meta['audio_id'].isin(audio_ids)]
    
#     if len(items_meta) == 0:
#         return None
    
#     # Most common language
#     lang_counts = items_meta['language'].value_counts()
#     preferred_language = lang_counts.index[0] if len(lang_counts) > 0 else None
#     lang_weight = lang_counts.iloc[0] / len(items_meta) if len(lang_counts) > 0 else 0
    
#     # Most common category
#     cat_counts = items_meta['category'].value_counts()
#     preferred_category = cat_counts.index[0] if len(cat_counts) > 0 else None
#     cat_weight = cat_counts.iloc[0] / len(items_meta) if len(cat_counts) > 0 else 0
    
#     return {
#         'preferred_language': preferred_language,
#         'language_weight': lang_weight,
#         'preferred_category': preferred_category,
#         'category_weight': cat_weight
#     }


# def recommend_for_user(user_id, top_k=10):
#     """
#     IMPROVED: Generate top-K recommendations with preference boosting
    
#     Args:
#         user_id: User ID (original ID, not encoded)
#         top_k: Number of recommendations to return
    
#     Returns:
#         List of recommended audio IDs
#     """
    
#     model, user_encoder, item_encoder, item_metadata, df = load_model_and_encoders()
    
#     # Load audio metadata
#     audio_meta = pd.read_csv("data/processed/audio_language_category.csv")
    
#     # Check if user exists in training data
#     if user_id not in user_encoder.classes_:
#         print(f"  User '{user_id}' not found in training data (cold start)")
#         print(f"  Returning {top_k} most popular items...\n")
        
#         # Return popular items (by average score)
#         popular_items = df.groupby('item_idx')['score'].mean().sort_values(ascending=False).head(top_k)
#         popular_audio_ids = item_encoder.inverse_transform(popular_items.index.values)
        
#         return popular_audio_ids.tolist()
    
#     # Encode user
#     user_idx = user_encoder.transform([user_id])[0]
    
#     # NEW: Get user preferences
#     user_prefs = get_user_preferences(user_idx, df, audio_meta, item_encoder)
    
#     # Get items user has already interacted with
#     user_items = set(df[df['user_idx'] == user_idx]['item_idx'].values)
    
#     # Get all items
#     all_items = set(range(df['item_idx'].max() + 1))
    
#     # Candidate items (not yet seen by user)
#     candidate_items = list(all_items - user_items)
    
#     if len(candidate_items) == 0:
#         print(f"⚠ User '{user_id}' has already seen all items!")
#         return []
    
#     # Get metadata for candidate items
#     candidate_metadata = item_metadata[item_metadata['item_idx'].isin(candidate_items)].copy()
    
#     # Merge with audio metadata to get language/category names
#     candidate_metadata = candidate_metadata.merge(
#         audio_meta[['audio_id', 'language', 'category']], 
#         left_on=item_encoder.inverse_transform(candidate_metadata['item_idx'].values),
#         right_on='audio_id',
#         how='left'
#     )
    
#     # Prepare tensors
#     item_tensor = torch.tensor(candidate_metadata['item_idx'].values, dtype=torch.long).to(DEVICE)
#     user_tensor = torch.full((len(candidate_metadata),), user_idx, dtype=torch.long).to(DEVICE)
#     language_tensor = torch.tensor(candidate_metadata['language_idx'].values, dtype=torch.long).to(DEVICE)
#     category_tensor = torch.tensor(candidate_metadata['category_idx'].values, dtype=torch.long).to(DEVICE)
    
#     # Predict scores
#     with torch.no_grad():
#         base_scores = model(user_tensor, item_tensor, language_tensor, category_tensor).cpu().numpy()
    
#     # FIX 1: PREFERENCE BOOSTING
#     # Boost items that match user's preferred language/category
#     if user_prefs:
#         preference_boost = np.zeros_like(base_scores)
        
#         for idx, row in enumerate(candidate_metadata.itertuples()):
#             boost = 0.0
            
#             # Language match
#             if user_prefs['preferred_language'] and row.language == user_prefs['preferred_language']:
#                 boost += 2.0 * user_prefs['language_weight']
            
#             # Category match
#             if user_prefs['preferred_category'] and row.category == user_prefs['preferred_category']:
#                 boost += 1.5 * user_prefs['category_weight']
            
#             preference_boost[idx] = boost
        
#         # Normalize
#         if preference_boost.max() > 0:
#             preference_boost = preference_boost / preference_boost.max()
        
#         # Combine: 60% model + 40% preferences
#         base_scores = 0.6 * base_scores + 0.4 * preference_boost * 5.0
    
#     # FIX 2: DIVERSITY PENALTY
#     # Penalize very popular items
#     item_popularity = df['item_idx'].value_counts().to_dict()
#     max_popularity = max(item_popularity.values()) if item_popularity else 1
    
#     popularity_penalty = np.array([
#         item_popularity.get(item_idx, 0) / max_popularity 
#         for item_idx in candidate_metadata['item_idx'].values
#     ])
    
#     # Apply diversity penalty (reduce scores of popular items)
#     diversity_weight = 0.3  # Adjust between 0.0-0.5
#     base_scores = base_scores - (diversity_weight * popularity_penalty * 2.0)
    
#     # Get top K recommendations
#     top_k = min(top_k, len(base_scores))
#     top_indices = np.argsort(base_scores)[::-1][:top_k]
    
#     # Map back to item indices
#     recommended_item_indices = candidate_metadata.iloc[top_indices]['item_idx'].values
    
#     # Decode to original audio IDs
#     recommended_audio_ids = item_encoder.inverse_transform(recommended_item_indices)
    
#     return recommended_audio_ids.tolist()


# def recommend_with_details(user_id, top_k=10):
#     """
#     IMPROVED: Generate recommendations with detailed information
    
#     Returns:
#         DataFrame with audio_id, predicted_score, language, category, match_reason
#     """
    
#     model, user_encoder, item_encoder, item_metadata, df = load_model_and_encoders()
    
#     # Load audio metadata for titles
#     audio_meta = pd.read_csv("data/processed/audio_language_category.csv")
    
#     if user_id not in user_encoder.classes_:
#         print(f"⚠ User '{user_id}' not found (cold start)")
#         popular_items = df.groupby('item_idx')['score'].mean().sort_values(ascending=False).head(top_k)
#         popular_audio_ids = item_encoder.inverse_transform(popular_items.index.values)
        
#         results = audio_meta[audio_meta['audio_id'].isin(popular_audio_ids)].drop_duplicates('audio_id')
#         results = results.copy()
#         results['predicted_score'] = 'N/A (Popular)'
#         results['match_reason'] = 'Cold Start'
#         return results[['audio_id', 'title', 'language', 'category', 'predicted_score', 'match_reason']].head(top_k)
    
#     user_idx = user_encoder.transform([user_id])[0]
#     user_prefs = get_user_preferences(user_idx, df, audio_meta, item_encoder)
    
#     user_items = set(df[df['user_idx'] == user_idx]['item_idx'].values)
#     all_items = set(range(df['item_idx'].max() + 1))
#     candidate_items = list(all_items - user_items)
    
#     if len(candidate_items) == 0:
#         return pd.DataFrame()
    
#     candidate_metadata = item_metadata[item_metadata['item_idx'].isin(candidate_items)].copy()
#     candidate_metadata = candidate_metadata.merge(
#         audio_meta[['audio_id', 'language', 'category']], 
#         left_on=item_encoder.inverse_transform(candidate_metadata['item_idx'].values),
#         right_on='audio_id',
#         how='left'
#     )
    
#     item_tensor = torch.tensor(candidate_metadata['item_idx'].values, dtype=torch.long).to(DEVICE)
#     user_tensor = torch.full((len(candidate_metadata),), user_idx, dtype=torch.long).to(DEVICE)
#     language_tensor = torch.tensor(candidate_metadata['language_idx'].values, dtype=torch.long).to(DEVICE)
#     category_tensor = torch.tensor(candidate_metadata['category_idx'].values, dtype=torch.long).to(DEVICE)
    
#     with torch.no_grad():
#         base_scores = model(user_tensor, item_tensor, language_tensor, category_tensor).cpu().numpy()
    
#     # Apply preference boosting
#     if user_prefs:
#         preference_boost = np.zeros_like(base_scores)
        
#         for idx, row in enumerate(candidate_metadata.itertuples()):
#             boost = 0.0
            
#             if user_prefs['preferred_language'] and row.language == user_prefs['preferred_language']:
#                 boost += 2.0 * user_prefs['language_weight']
            
#             if user_prefs['preferred_category'] and row.category == user_prefs['preferred_category']:
#                 boost += 1.5 * user_prefs['category_weight']
            
#             preference_boost[idx] = boost
        
#         if preference_boost.max() > 0:
#             preference_boost = preference_boost / preference_boost.max()
        
#         base_scores = 0.6 * base_scores + 0.4 * preference_boost * 5.0
    
#     # Apply diversity penalty
#     item_popularity = df['item_idx'].value_counts().to_dict()
#     max_popularity = max(item_popularity.values()) if item_popularity else 1
    
#     popularity_penalty = np.array([
#         item_popularity.get(item_idx, 0) / max_popularity
#         for item_idx in candidate_metadata['item_idx'].values
#     ])
    
#     base_scores = base_scores - (0.3 * popularity_penalty * 2.0)
    
#     top_k = min(top_k, len(base_scores))
#     top_indices = np.argsort(base_scores)[::-1][:top_k]
#     top_scores = base_scores[top_indices]
    
#     recommended_item_indices = candidate_metadata.iloc[top_indices]['item_idx'].values
#     recommended_audio_ids = item_encoder.inverse_transform(recommended_item_indices)
    
#     # Create results DataFrame with match reasons
#     results = []
#     for audio_id, score, idx in zip(recommended_audio_ids, top_scores, top_indices):
#         audio_info = audio_meta[audio_meta['audio_id'] == audio_id].iloc[0]
        
#         # Determine why this was recommended
#         match_reasons = []
#         if user_prefs:
#             if user_prefs['preferred_language'] and audio_info['language'] == user_prefs['preferred_language']:
#                 match_reasons.append(f"Language: {audio_info['language']}")
#             if user_prefs['preferred_category'] and audio_info['category'] == user_prefs['preferred_category']:
#                 match_reasons.append(f"Category: {audio_info['category']}")
        
#         if not match_reasons:
#             match_reasons.append("Model Score")
        
#         results.append({
#             'audio_id': audio_id,
#             'title': audio_info['title'],
#             'language': audio_info['language'],
#             'category': audio_info['category'],
#             'predicted_score': f"{score:.2f}",
#             'match_reason': " + ".join(match_reasons)
#         })
    
#     return pd.DataFrame(results)


# # Example usage
# if __name__ == "__main__":
#     print("\n" + "="*80)
#     print("IMPROVED RECOMMENDATION SYSTEM - TESTING")
#     print("="*80 + "\n")
    
#     # Load a sample user
#     df = pd.read_csv(DATA_PATH)
    
#     with open(USER_ENCODER_PATH, "rb") as f:
#         user_encoder = pickle.load(f)
    
#     # Get first 3 users
#     sample_users = user_encoder.inverse_transform(range(min(3, len(user_encoder.classes_))))
    
#     for user_id in sample_users:
#         print(f"Recommendations for User: {user_id}")
#         print("-"*80)
        
#         recs = recommend_with_details(user_id, top_k=5)
        
#         if len(recs) > 0:
#             for i, row in recs.iterrows():
#                 print(f"{i+1}. {row['title'][:50]}")
#                 print(f"   Audio ID: {row['audio_id']}")
#                 print(f"   Language: {row['language']} | Category: {row['category']}")
#                 print(f"   Predicted Score: {row['predicted_score']}")
#                 print(f"   Why: {row['match_reason']}")
#                 print()
#         else:
#             print("  No recommendations available\n")
    
"""
RECOMMENDATION WITH FIXED DIVERSITY
Changes:
- Stronger popularity penalty (0.5 instead of 0.3)
- Exponential penalty for very popular items
- Added randomization for equal-scored items
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.ncf_model import ImprovedNCF
import pickle
import pandas as pd
import torch
import numpy as np

MODEL_PATH = PROJECT_ROOT / "models_saved/recommender_improved.pt"
DATA_PATH = PROJECT_ROOT / "data/processed/ml_interactions.csv"
ITEM_METADATA_PATH = PROJECT_ROOT / "data/processed/item_metadata.csv"
AUDIO_META_PATH = PROJECT_ROOT / "data/processed/audio_language_category.csv"

USER_ENCODER_PATH = PROJECT_ROOT / "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = PROJECT_ROOT / "models_saved/item_encoder.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_encoders():
    """Load trained model and encoders"""
    df = pd.read_csv(DATA_PATH)
    
    num_users = df["user_idx"].max() + 1
    num_items = df["item_idx"].max() + 1
    num_languages = df["language_idx"].max() + 1
    num_categories = df["category_idx"].max() + 1
    
    model = ImprovedNCF(
        num_users=num_users,
        num_items=num_items,
        num_languages=num_languages,
        num_categories=num_categories,
        embedding_dim=64
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)
    
    item_metadata = pd.read_csv(ITEM_METADATA_PATH)
    
    return model, user_encoder, item_encoder, item_metadata, df


def get_user_preferences(user_idx, df, audio_meta, item_encoder):
    """Extract user's language and category preferences"""
    user_data = df[df['user_idx'] == user_idx]
    
    if len(user_data) == 0:
        return None
    
    liked_items = user_data[user_data['score'] >= 4.0]
    
    if len(liked_items) < 2:
        liked_items = user_data
    
    audio_ids = item_encoder.inverse_transform(liked_items['item_idx'].values)
    items_meta = audio_meta[audio_meta['audio_id'].isin(audio_ids)]
    
    if len(items_meta) == 0:
        return None
    
    lang_counts = items_meta['language'].value_counts()
    preferred_language = lang_counts.index[0] if len(lang_counts) > 0 else None
    lang_weight = lang_counts.iloc[0] / len(items_meta) if len(lang_counts) > 0 else 0
    
    cat_counts = items_meta['category'].value_counts()
    preferred_category = cat_counts.index[0] if len(cat_counts) > 0 else None
    cat_weight = cat_counts.iloc[0] / len(items_meta) if len(cat_counts) > 0 else 0
    
    return {
        'preferred_language': preferred_language,
        'language_weight': lang_weight,
        'preferred_category': preferred_category,
        'category_weight': cat_weight
    }


def recommend_for_user(user_id, top_k=10):
    """
    Generate top-K recommendations with FIXED DIVERSITY
    
    Key changes:
    - Stronger popularity penalty (0.5 → 0.7)
    - Exponential penalty for very popular items
    - Random tie-breaking
    """
    
    model, user_encoder, item_encoder, item_metadata, df = load_model_and_encoders()
    audio_meta = pd.read_csv(AUDIO_META_PATH)
    
    # Handle unknown users
    if user_id not in user_encoder.classes_:
        print(f"⚠ User '{user_id}' not found in training data (cold start)")
        print(f"  Returning {top_k} most popular items...\n")
        
        popular_items = df.groupby('item_idx')['score'].mean().sort_values(ascending=False).head(top_k)
        popular_audio_ids = item_encoder.inverse_transform(popular_items.index.values)
        
        return popular_audio_ids.tolist()
    
    user_idx = user_encoder.transform([user_id])[0]
    user_prefs = get_user_preferences(user_idx, df, audio_meta, item_encoder)
    
    # Get candidate items
    user_items = set(df[df['user_idx'] == user_idx]['item_idx'].values)
    all_items = set(range(df['item_idx'].max() + 1))
    candidate_items = list(all_items - user_items)
    
    if len(candidate_items) == 0:
        print(f"⚠ User '{user_id}' has already seen all items!")
        return []
    
    candidate_metadata = item_metadata[item_metadata['item_idx'].isin(candidate_items)].copy()
    candidate_metadata = candidate_metadata.merge(
        audio_meta[['audio_id', 'language', 'category']], 
        left_on=item_encoder.inverse_transform(candidate_metadata['item_idx'].values),
        right_on='audio_id',
        how='left'
    )
    
    # Get model predictions
    item_tensor = torch.tensor(candidate_metadata['item_idx'].values, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((len(candidate_metadata),), user_idx, dtype=torch.long).to(DEVICE)
    language_tensor = torch.tensor(candidate_metadata['language_idx'].values, dtype=torch.long).to(DEVICE)
    category_tensor = torch.tensor(candidate_metadata['category_idx'].values, dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        base_scores = model(user_tensor, item_tensor, language_tensor, category_tensor).cpu().numpy()
    
    # COMPONENT 1: Preference Boosting (keep as is - working great!)
    if user_prefs:
        preference_boost = np.zeros_like(base_scores)
        
        for idx, row in enumerate(candidate_metadata.itertuples()):
            boost = 0.0
            
            if user_prefs['preferred_language'] and row.language == user_prefs['preferred_language']:
                boost += 2.0 * user_prefs['language_weight']
            
            if user_prefs['preferred_category'] and row.category == user_prefs['preferred_category']:
                boost += 1.5 * user_prefs['category_weight']
            
            preference_boost[idx] = boost
        
        if preference_boost.max() > 0:
            preference_boost = preference_boost / preference_boost.max()
        
        base_scores = 0.6 * base_scores + 0.4 * preference_boost * 5.0
    
    # COMPONENT 2: FIXED - Stronger Diversity Penalty
    item_popularity = df['item_idx'].value_counts().to_dict()
    max_popularity = max(item_popularity.values()) if item_popularity else 1
    
    popularity_penalty = np.array([
        item_popularity.get(item_idx, 0) / max_popularity
        for item_idx in candidate_metadata['item_idx'].values
    ])
    
    # CRITICAL FIX: Use EXPONENTIAL penalty for very popular items
    # This heavily penalizes the top 20 most popular items
    popularity_penalty_exp = popularity_penalty ** 2  # Square it = stronger penalty
    
    # CRITICAL FIX: Increase penalty weight from 0.3 to 0.7
    diversity_weight = 0.7  # ← CHANGED from 0.3
    base_scores = base_scores - (diversity_weight * popularity_penalty_exp * 3.0)  # ← Also increased multiplier
    
    # COMPONENT 3: Add small random noise for tie-breaking
    # This prevents always recommending same items when scores are equal
    random_noise = np.random.uniform(-0.1, 0.1, size=len(base_scores))
    base_scores = base_scores + random_noise
    
    # Get top K recommendations
    top_k = min(top_k, len(base_scores))
    top_indices = np.argsort(base_scores)[::-1][:top_k]
    
    recommended_item_indices = candidate_metadata.iloc[top_indices]['item_idx'].values
    recommended_audio_ids = item_encoder.inverse_transform(recommended_item_indices)
    
    return recommended_audio_ids.tolist()


def recommend_with_details(user_id, top_k=10):
    """
    Generate recommendations with detailed information and FIXED DIVERSITY
    """
    
    model, user_encoder, item_encoder, item_metadata, df = load_model_and_encoders()
    audio_meta = pd.read_csv(AUDIO_META_PATH)

    
    if user_id not in user_encoder.classes_:
        print(f"⚠ User '{user_id}' not found (cold start)")
        popular_items = df.groupby('item_idx')['score'].mean().sort_values(ascending=False).head(top_k)
        popular_audio_ids = item_encoder.inverse_transform(popular_items.index.values)
        
        results = audio_meta[audio_meta['audio_id'].isin(popular_audio_ids)].drop_duplicates('audio_id')
        results = results.copy()
        results['predicted_score'] = 'N/A (Popular)'
        results['match_reason'] = 'Cold Start'
        return results[['audio_id', 'title', 'language', 'category', 'predicted_score', 'match_reason']].head(top_k)
    
    user_idx = user_encoder.transform([user_id])[0]
    user_prefs = get_user_preferences(user_idx, df, audio_meta, item_encoder)
    
    user_items = set(df[df['user_idx'] == user_idx]['item_idx'].values)
    all_items = set(range(df['item_idx'].max() + 1))
    candidate_items = list(all_items - user_items)
    
    if len(candidate_items) == 0:
        return pd.DataFrame()
    
    candidate_metadata = item_metadata[item_metadata['item_idx'].isin(candidate_items)].copy()
    candidate_metadata = candidate_metadata.merge(
        audio_meta[['audio_id', 'language', 'category']], 
        left_on=item_encoder.inverse_transform(candidate_metadata['item_idx'].values),
        right_on='audio_id',
        how='left'
    )
    
    item_tensor = torch.tensor(candidate_metadata['item_idx'].values, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((len(candidate_metadata),), user_idx, dtype=torch.long).to(DEVICE)
    language_tensor = torch.tensor(candidate_metadata['language_idx'].values, dtype=torch.long).to(DEVICE)
    category_tensor = torch.tensor(candidate_metadata['category_idx'].values, dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        base_scores = model(user_tensor, item_tensor, language_tensor, category_tensor).cpu().numpy()
    
    # Apply preference boosting
    if user_prefs:
        preference_boost = np.zeros_like(base_scores)
        
        for idx, row in enumerate(candidate_metadata.itertuples()):
            boost = 0.0
            
            if user_prefs['preferred_language'] and row.language == user_prefs['preferred_language']:
                boost += 2.0 * user_prefs['language_weight']
            
            if user_prefs['preferred_category'] and row.category == user_prefs['preferred_category']:
                boost += 1.5 * user_prefs['category_weight']
            
            preference_boost[idx] = boost
        
        if preference_boost.max() > 0:
            preference_boost = preference_boost / preference_boost.max()
        
        base_scores = 0.6 * base_scores + 0.4 * preference_boost * 5.0
    
    # Apply FIXED diversity penalty
    item_popularity = df['item_idx'].value_counts().to_dict()
    max_popularity = max(item_popularity.values()) if item_popularity else 1
    
    popularity_penalty = np.array([
        item_popularity.get(item_idx, 0) / max_popularity
        for item_idx in candidate_metadata['item_idx'].values
    ])
    
    popularity_penalty_exp = popularity_penalty ** 2
    base_scores = base_scores - (0.7 * popularity_penalty_exp * 3.0)
    
    # Add random noise
    random_noise = np.random.uniform(-0.1, 0.1, size=len(base_scores))
    base_scores = base_scores + random_noise
    
    top_k = min(top_k, len(base_scores))
    top_indices = np.argsort(base_scores)[::-1][:top_k]
    top_scores = base_scores[top_indices]
    
    recommended_item_indices = candidate_metadata.iloc[top_indices]['item_idx'].values
    recommended_audio_ids = item_encoder.inverse_transform(recommended_item_indices)
    
    # Create results DataFrame with match reasons
    results = []
    for audio_id, score, idx in zip(recommended_audio_ids, top_scores, top_indices):
        audio_info = audio_meta[audio_meta['audio_id'] == audio_id].iloc[0]
        
        match_reasons = []
        if user_prefs:
            if user_prefs['preferred_language'] and audio_info['language'] == user_prefs['preferred_language']:
                match_reasons.append(f"Language: {audio_info['language']}")
            if user_prefs['preferred_category'] and audio_info['category'] == user_prefs['preferred_category']:
                match_reasons.append(f"Category: {audio_info['category']}")
        
        if not match_reasons:
            match_reasons.append("Model Score")
        
        results.append({
            'audio_id': audio_id,
            'title': audio_info['title'],
            'language': audio_info['language'],
            'category': audio_info['category'],
            'predicted_score': f"{score:.2f}",
            'match_reason': " + ".join(match_reasons)
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING FIXED DIVERSITY")
    print("="*80 + "\n")
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    # Test diversity
    num_test_users = min(10, len(user_encoder.classes_))
    all_recommendations = []
    
    for i in range(num_test_users):
        user_id = user_encoder.inverse_transform([i])[0]
        recs = recommend_for_user(user_id, top_k=10)
        all_recommendations.extend(recs)
    
    unique_items = len(set(all_recommendations))
    total_recs = len(all_recommendations)
    diversity = unique_items / total_recs * 100 if total_recs > 0 else 0
    
    print(f"Diversity Test Results:")
    print(f"  Total recommendations: {total_recs}")
    print(f"  Unique items: {unique_items}")
    print(f"  Diversity score: {diversity:.1f}%")
    print()
    
    if diversity > 50:
        print("✅ EXCELLENT - Diversity significantly improved!")
    elif diversity > 35:
        print("✅ GOOD - Diversity improved!")
    else:
        print("⚠ Still needs work - Consider data balancing")
    
    print("\n" + "="*80 + "\n")