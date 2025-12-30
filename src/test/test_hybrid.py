import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
import pickle

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
    """
    Hybrid recommender that combines:
    1. Content-based filtering (by language/category)
    2. Neural network ranking (for personalization)
    """
    
    def __init__(self):
        print("\n" + "="*70)
        print("LOADING HYBRID RECOMMENDER")
        print("="*70 + "\n")
        
        # Load model
        self._load_model()
        
        # Load audio metadata
        self.audio_df = pd.read_csv(AUDIO_DATA_PATH)
        print(f"✓ Loaded {len(self.audio_df)} audio metadata records")
        
        # Build item lookup
        self._build_item_lookup()
    
    def _load_model(self):
        """Load neural network model and encoders"""
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
        
        print(f"✓ Loaded model and encoders")
    
    def _build_item_lookup(self):
        """Build lookup: item_id -> {language, category}"""
        self.item_metadata = {}
        
        # Group by audio_id, take first entry
        grouped = self.audio_df.groupby('audio_id').first().reset_index()
        
        for _, row in grouped.iterrows():
            audio_id = row['audio_id']
            if audio_id in self.item_encoder.classes_:
                item_idx = self.item_encoder.transform([audio_id])[0]
                self.item_metadata[item_idx] = {
                    'audio_id': audio_id,
                    'language': row['language'],
                    'category': row['category']
                }
        
        print(f"✓ Built metadata for {len(self.item_metadata)} items\n")
    
    def recommend(self, user_id, language, category, top_k=10):
        """
        Recommend items for a user, filtered by language and category
        
        Args:
            user_id: User ID
            language: Language to filter by
            category: Category to filter by
            top_k: Number of recommendations
            
        Returns:
            List of audio IDs
        """
        print(f"\nGenerating recommendations...")
        print(f"  User: {user_id}")
        print(f"  Filter: {language} → {category}")
        
        # Step 1: Filter items by language and category
        filtered_item_indices = []
        for item_idx, metadata in self.item_metadata.items():
            if metadata['language'] == language and metadata['category'] == category:
                filtered_item_indices.append(item_idx)
        
        print(f"✓ Filtered to {len(filtered_item_indices)} items")
        
        if len(filtered_item_indices) == 0:
            print("⚠ No items match this language-category combination")
            return []
        
        # Step 2: Get user index
        if user_id not in self.user_encoder.classes_:
            # Use first user as default
            user_id = self.user_encoder.classes_[0]
            print(f"⚠ User not found, using default: {user_id}")
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Step 3: Get language and category indices
        language_idx = self.language_encoder.transform([language])[0]
        category_idx = self.category_encoder.transform([category])[0]
        
        # Step 4: Score the FILTERED items using the neural network
        item_tensor = torch.tensor(filtered_item_indices, dtype=torch.long).to(DEVICE)
        user_tensor = torch.full((len(filtered_item_indices),), user_idx, dtype=torch.long).to(DEVICE)
        language_tensor = torch.full((len(filtered_item_indices),), language_idx, dtype=torch.long).to(DEVICE)
        category_tensor = torch.full((len(filtered_item_indices),), category_idx, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor, language_tensor, category_tensor)
        
        # Step 5: Get top-K
        if len(scores) <= top_k:
            top_indices = torch.arange(len(scores))
        else:
            top_indices = torch.topk(scores, top_k).indices
        
        top_item_indices = [filtered_item_indices[i] for i in top_indices.cpu().numpy()]
        recommended_audio_ids = self.item_encoder.inverse_transform(top_item_indices)
        
        print(f"✓ Generated {len(recommended_audio_ids)} recommendations\n")
        
        return list(recommended_audio_ids)


# ---------------- SAVE AS NEW recommend.py ----------------
def save_for_production():
    """
    Generate the recommend.py file for production use
    """
    code = '''"""
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
'''
    
    with open("src/inference/recommend_hybrid.py", "w") as f:
        f.write(code)
    
    print("✓ Saved production code to: src/inference/recommend_hybrid.py\n")


# ---------------- TEST ----------------
if __name__ == "__main__":
    recommender = HybridRecommender()
    
    print("="*70)
    print("TESTING HYBRID RECOMMENDER")
    print("="*70)
    
    # Get a user
    user_id = recommender.user_encoder.classes_[0]
    
    # Test cases
    test_cases = [
        ("English", "Psychology"),
        ("Hindi", "Health"),
        ("Marathi", "Spiritual"),
    ]
    
    for language, category in test_cases:
        recommendations = recommender.recommend(user_id, language, category, top_k=10)
        
        print(f"{'─'*70}")
        print(f"Recommendations for: {language} → {category}")
        print(f"{'─'*70}")
        for i, audio_id in enumerate(recommendations[:5], 1):
            print(f"  {i}. {audio_id}")
        if len(recommendations) > 5:
            print(f"  ... and {len(recommendations) - 5} more")
        print()
    
    # Save production code
    save_for_production()