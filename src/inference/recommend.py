import numpy as np
import torch
import pandas as pd
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.models.ncf_model import ImprovedNCF

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


def get_user_preferences(user_idx, df, audio_meta, item_encoder, top_n_prefs=3):
    user_data = df[df['user_idx'] == user_idx]

    if len(user_data) == 0:
        return None

    # Focus on highly-rated items (score >= 4)
    liked_items = user_data[user_data['score'] >= 4.0]

    # Fallback: if too few likes, use all interactions
    if len(liked_items) < 2:
        liked_items = user_data

    # Get audio metadata for liked items
    audio_ids = item_encoder.inverse_transform(liked_items['item_idx'].values)
    items_meta = audio_meta[audio_meta['audio_id'].isin(audio_ids)]

    if len(items_meta) == 0:
        return None

    # Get TOP N languages with their weights
    lang_counts = items_meta['language'].value_counts()
    top_languages = []

    for i, (lang, count) in enumerate(lang_counts.head(top_n_prefs).items()):
        weight = count / len(items_meta)
        top_languages.append({
            'language': lang,
            'weight': weight,
            'rank': i + 1  # 1st, 2nd, 3rd preference
        })

    # Get TOP N categories with their weights
    cat_counts = items_meta['category'].value_counts()
    top_categories = []

    for i, (cat, count) in enumerate(cat_counts.head(top_n_prefs).items()):
        weight = count / len(items_meta)
        top_categories.append({
            'category': cat,
            'weight': weight,
            'rank': i + 1
        })

    return {
        'preferred_languages': top_languages,
        'preferred_categories': top_categories,
        'total_interactions': len(items_meta)
    }


def calculate_preference_boost(audio_info, user_prefs, exploration_rate=0.15):
    """
    FIXED: Flattened boosting to ensure secondary/tertiary preferences compete better
    """
    if not user_prefs:
        return 0.0

    boost = 0.0
    language_matched = False
    category_matched = False

    # IMPROVED: Flattened rank multipliers for better diversity
    # Primary: 1.0, Secondary: 0.85, Tertiary: 0.70 (much closer now!)
    for pref in user_prefs['preferred_languages']:
        if audio_info['language'] == pref['language']:
            rank_multiplier = 1.0 if pref['rank'] == 1 else (
                0.85 if pref['rank'] == 2 else 0.70)
            boost += 1.8 * rank_multiplier  # Reduced base boost
            language_matched = True
            break

    # IMPROVED: Flattened category multipliers
    for pref in user_prefs['preferred_categories']:
        if audio_info['category'] == pref['category']:
            rank_multiplier = 1.0 if pref['rank'] == 1 else (
                0.85 if pref['rank'] == 2 else 0.70)
            boost += 1.5 * rank_multiplier  # Reduced base boost
            category_matched = True
            break

    # IMPROVED EXPLORATION LOGIC
    # 15% chance to boost ANY item (even non-matching) for true diversity
    if np.random.random() < exploration_rate:
        boost += 0.8  # Add exploration boost to ALL items with 15% probability

    return boost


def recommend_for_user(user_id, top_k=10, exploration_rate=0.15, session_id=None, enable_randomness=True):

    model, user_encoder, item_encoder, item_metadata, df = load_model_and_encoders()
    audio_meta = pd.read_csv(AUDIO_META_PATH)

    # Cold start handling
    if user_id not in user_encoder.classes_:
        print(f"User '{user_id}' not found (cold start)")
        print(f"Returning {top_k} popular items...\n")

        popular_items = df.groupby('item_idx')['score'].mean().sort_values(
            ascending=False).head(top_k)
        popular_audio_ids = item_encoder.inverse_transform(
            popular_items.index.values)
        return popular_audio_ids.tolist()

    user_idx = user_encoder.transform([user_id])[0]

    # Get multi-preferences
    user_prefs = get_user_preferences(
        user_idx, df, audio_meta, item_encoder, top_n_prefs=3)

    # Get candidate items
    user_items = set(df[df['user_idx'] == user_idx]['item_idx'].values)
    all_items = set(range(df['item_idx'].max() + 1))
    candidate_items = list(all_items - user_items)

    if len(candidate_items) == 0:
        print(f"User '{user_id}' has seen all items!")
        return []

    # Prepare candidate metadata
    candidate_metadata = item_metadata[item_metadata['item_idx'].isin(
        candidate_items)].copy()
    candidate_metadata = candidate_metadata.merge(
        audio_meta[['audio_id', 'language', 'category']],
        left_on=item_encoder.inverse_transform(
            candidate_metadata['item_idx'].values),
        right_on='audio_id',
        how='left'
    )

    # Model predictions
    item_tensor = torch.tensor(
        candidate_metadata['item_idx'].values, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((len(candidate_metadata),),
                             user_idx, dtype=torch.long).to(DEVICE)
    language_tensor = torch.tensor(
        candidate_metadata['language_idx'].values, dtype=torch.long).to(DEVICE)
    category_tensor = torch.tensor(
        candidate_metadata['category_idx'].values, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        base_scores = model(user_tensor, item_tensor,
                            language_tensor, category_tensor).cpu().numpy()

    # IMPROVED: Multi-preference boosting with proper exploration
    if user_prefs:
        preference_boost = np.zeros(len(candidate_metadata))

        for idx, row in enumerate(candidate_metadata.itertuples()):
            audio_info = {
                'language': row.language,
                'category': row.category
            }
            preference_boost[idx] = calculate_preference_boost(
                audio_info, user_prefs, exploration_rate
            )

        # Normalize
        if preference_boost.max() > 0:
            preference_boost = preference_boost / preference_boost.max()

        # IMPROVED: Balanced weighting to allow all preferences to compete
        # Model predictions still matter, but preferences get fair representation
        base_scores = 0.45 * base_scores + 0.55 * preference_boost * 4.5

    # Diversity penalty (unchanged - working well)
    item_popularity = df['item_idx'].value_counts().to_dict()
    max_popularity = max(item_popularity.values()) if item_popularity else 1

    popularity_penalty = np.array([
        item_popularity.get(item_idx, 0) / max_popularity
        for item_idx in candidate_metadata['item_idx'].values
    ])

    popularity_penalty_exp = popularity_penalty ** 2
    base_scores = base_scores - (0.7 * popularity_penalty_exp * 3.0)

    # Controlled randomness for tie-breaking
    if enable_randomness:
        # Set seed based on user_id + session_id for consistency within sessions
        if session_id is not None:
            # Same session = same recommendations
            seed = hash(f"{user_id}_{session_id}") % (2**32)
        else:
            # Different recommendations each time (current behavior)
            seed = None

        # Set numpy random seed
        if seed is not None:
            np.random.seed(seed)

        # INCREASED randomness to help break ties between preference tiers
        random_noise = np.random.uniform(-0.2, 0.2, size=len(base_scores))
        base_scores = base_scores + random_noise

        # Reset seed to avoid affecting other operations
        if seed is not None:
            np.random.seed(None)

    # Get top K
    top_k = min(top_k, len(base_scores))
    top_indices = np.argsort(base_scores)[::-1][:top_k]

    recommended_item_indices = candidate_metadata.iloc[top_indices]['item_idx'].values
    recommended_audio_ids = item_encoder.inverse_transform(
        recommended_item_indices)

    return recommended_audio_ids.tolist()


def recommend_with_details(user_id, top_k=10, exploration_rate=0.15, session_id=None, enable_randomness=True):
    model, user_encoder, item_encoder, item_metadata, df = load_model_and_encoders()
    audio_meta = pd.read_csv(AUDIO_META_PATH)

    # Cold start
    if user_id not in user_encoder.classes_:
        print(f"User '{user_id}' not found (cold start)")
        popular_items = df.groupby('item_idx')['score'].mean().sort_values(
            ascending=False).head(top_k)
        popular_audio_ids = item_encoder.inverse_transform(
            popular_items.index.values)

        results = audio_meta[audio_meta['audio_id'].isin(
            popular_audio_ids)].drop_duplicates('audio_id')
        results = results.copy()
        results['predicted_score'] = 'N/A'
        results['match_reason'] = 'Popular (Cold Start)'
        return results[['audio_id', 'title', 'language', 'category', 'predicted_score', 'match_reason']].head(top_k)

    user_idx = user_encoder.transform([user_id])[0]
    user_prefs = get_user_preferences(
        user_idx, df, audio_meta, item_encoder, top_n_prefs=3)

    # Get candidates
    user_items = set(df[df['user_idx'] == user_idx]['item_idx'].values)
    all_items = set(range(df['item_idx'].max() + 1))
    candidate_items = list(all_items - user_items)

    if len(candidate_items) == 0:
        return pd.DataFrame()

    candidate_metadata = item_metadata[item_metadata['item_idx'].isin(
        candidate_items)].copy()
    candidate_metadata = candidate_metadata.merge(
        audio_meta[['audio_id', 'language', 'category']],
        left_on=item_encoder.inverse_transform(
            candidate_metadata['item_idx'].values),
        right_on='audio_id',
        how='left'
    )

    # Model predictions
    item_tensor = torch.tensor(
        candidate_metadata['item_idx'].values, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((len(candidate_metadata),),
                             user_idx, dtype=torch.long).to(DEVICE)
    language_tensor = torch.tensor(
        candidate_metadata['language_idx'].values, dtype=torch.long).to(DEVICE)
    category_tensor = torch.tensor(
        candidate_metadata['category_idx'].values, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        base_scores = model(user_tensor, item_tensor,
                            language_tensor, category_tensor).cpu().numpy()

    # Multi-preference boosting with proper exploration
    if user_prefs:
        preference_boost = np.zeros(len(candidate_metadata))

        for idx, row in enumerate(candidate_metadata.itertuples()):
            audio_info = {'language': row.language, 'category': row.category}
            preference_boost[idx] = calculate_preference_boost(
                audio_info, user_prefs, exploration_rate
            )

        if preference_boost.max() > 0:
            preference_boost = preference_boost / preference_boost.max()

        # IMPROVED: Balanced weighting
        base_scores = 0.45 * base_scores + 0.55 * preference_boost * 4.5

    # Diversity penalty
    item_popularity = df['item_idx'].value_counts().to_dict()
    max_popularity = max(item_popularity.values()) if item_popularity else 1

    popularity_penalty = np.array([
        item_popularity.get(item_idx, 0) / max_popularity
        for item_idx in candidate_metadata['item_idx'].values
    ])

    popularity_penalty_exp = popularity_penalty ** 2
    base_scores = base_scores - (0.7 * popularity_penalty_exp * 3.0)

    # Controlled randomness
    if enable_randomness:
        if session_id is not None:
            seed = hash(f"{user_id}_{session_id}") % (2**32)
            np.random.seed(seed)

        random_noise = np.random.uniform(-0.2, 0.2, size=len(base_scores))
        base_scores = base_scores + random_noise

        if session_id is not None:
            np.random.seed(None)

    # Get top K
    top_k = min(top_k, len(base_scores))
    top_indices = np.argsort(base_scores)[::-1][:top_k]
    top_scores = base_scores[top_indices]

    recommended_item_indices = candidate_metadata.iloc[top_indices]['item_idx'].values
    recommended_audio_ids = item_encoder.inverse_transform(
        recommended_item_indices)

    # Build results with detailed match reasons
    results = []
    for audio_id, score, idx in zip(recommended_audio_ids, top_scores, top_indices):
        audio_info = audio_meta[audio_meta['audio_id'] == audio_id].iloc[0]

        match_reasons = []
        language_matched = False
        category_matched = False

        if user_prefs:
            # Check language matches
            for pref in user_prefs['preferred_languages']:
                if audio_info['language'] == pref['language']:
                    rank_label = {1: "Primary", 2: "Secondary",
                                  3: "Tertiary"}[pref['rank']]
                    match_reasons.append(
                        f"{rank_label} Language: {pref['language']}")
                    language_matched = True
                    break

            # Check category matches
            for pref in user_prefs['preferred_categories']:
                if audio_info['category'] == pref['category']:
                    rank_label = {1: "Primary", 2: "Secondary",
                                  3: "Tertiary"}[pref['rank']]
                    match_reasons.append(
                        f"{rank_label} Category: {pref['category']}")
                    category_matched = True
                    break

        # FIXED: Better exploration detection
        if not match_reasons:
            match_reasons.append("Exploration / Model Score")

        results.append({
            'audio_id': audio_id,
            'title': audio_info['title'],
            'language': audio_info['language'],
            'category': audio_info['category'],
            'predicted_score': f"{score:.2f}",
            'match_reason': " + ".join(match_reasons)
        })

    return pd.DataFrame(results)
