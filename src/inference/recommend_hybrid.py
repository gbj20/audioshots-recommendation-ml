"""
Hybrid Recommendation System for Production
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
import pickle
from src.models.ncf_model import NCF

MODEL_PATH = "models_saved/recommender.pt"
DATA_PATH = "data/processed/ml_interactions.csv"
AUDIO_DATA_PATH = "data/processed/audio_language_category.csv"
USER_ENCODER_PATH = "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"
LANGUAGE_ENCODER_PATH = "models_saved/language_encoder.pkl"
CATEGORY_ENCODER_PATH = "models_saved/category_encoder.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for caching
_model = None
_item_metadata = None
_encoders = None

def load_model():
    """Load model and metadata (cached)"""
    global _model, _item_metadata, _encoders
    
    if _model is not None:
        return _model, _item_metadata, _encoders
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    audio_df = pd.read_csv(AUDIO_DATA_PATH)
    
    # Load encoders
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)
    with open(LANGUAGE_ENCODER_PATH, "rb") as f:
        language_encoder = pickle.load(f)
    with open(CATEGORY_ENCODER_PATH, "rb") as f:
        category_encoder = pickle.load(f)
    
    # Load model
    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()
    num_languages = df["language_idx"].nunique()
    num_categories = df["category_idx"].nunique()
    
    model = NCF(num_users, num_items, num_languages, num_categories)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Build item metadata
    item_metadata = {}
    grouped = audio_df.groupby('audio_id').first().reset_index()
    for _, row in grouped.iterrows():
        audio_id = row['audio_id']
        if audio_id in item_encoder.classes_:
            item_idx = item_encoder.transform([audio_id])[0]
            item_metadata[item_idx] = {
                'language': row['language'],
                'category': row['category']
            }
    
    _model = model
    _item_metadata = item_metadata
    _encoders = {
        'user': user_encoder,
        'item': item_encoder,
        'language': language_encoder,
        'category': category_encoder
    }
    
    return model, item_metadata, _encoders


def recommend_for_user(user_id, language=None, category=None, top_k=10):
    """
    Recommend items for a user
    
    Args:
        user_id: User ID
        language: Optional language filter
        category: Optional category filter
        top_k: Number of recommendations
        
    Returns:
        List of audio IDs
    """
    model, item_metadata, encoders = load_model()
    
    user_encoder = encoders['user']
    item_encoder = encoders['item']
    language_encoder = encoders['language']
    category_encoder = encoders['category']
    
    # Get user index
    if user_id not in user_encoder.classes_:
        raise ValueError(f"User {user_id} not found")
    
    user_idx = user_encoder.transform([user_id])[0]
    
    # Filter items by language and category if specified
    if language and category:
        filtered_items = [
            idx for idx, meta in item_metadata.items()
            if meta['language'] == language and meta['category'] == category
        ]
    elif language:
        filtered_items = [
            idx for idx, meta in item_metadata.items()
            if meta['language'] == language
        ]
    elif category:
        filtered_items = [
            idx for idx, meta in item_metadata.items()
            if meta['category'] == category
        ]
    else:
        filtered_items = list(item_metadata.keys())
    
    if not filtered_items:
        return []
    
    # Get language and category for scoring
    if language:
        lang_idx = language_encoder.transform([language])[0]
    else:
        lang_idx = 0  # Default
    
    if category:
        cat_idx = category_encoder.transform([category])[0]
    else:
        cat_idx = 0  # Default
    
    # Score items
    item_tensor = torch.tensor(filtered_items, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((len(filtered_items),), user_idx, dtype=torch.long).to(DEVICE)
    lang_tensor = torch.full((len(filtered_items),), lang_idx, dtype=torch.long).to(DEVICE)
    cat_tensor = torch.full((len(filtered_items),), cat_idx, dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        scores = model(user_tensor, item_tensor, lang_tensor, cat_tensor)
    
    # Get top-K
    k = min(top_k, len(scores))
    top_indices = torch.topk(scores, k).indices.cpu().numpy()
    top_items = [filtered_items[i] for i in top_indices]
    
    return item_encoder.inverse_transform(top_items)
