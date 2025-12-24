import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import pandas as pd
import pickle

from src.models.ncf_model import NCF

# --------------------------------
# CONFIG
# --------------------------------
MODEL_PATH = "models_saved/recommender.pt"
DATA_PATH = "data/processed/ml_interactions.csv"
USER_ENCODER_PATH = "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"
LANGUAGE_ENCODER_PATH = "models_saved/language_encoder.pkl"
USER_LANGUAGE_MAP_PATH = "data/processed/user_language_map.csv"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------
# LOAD MODEL & ENCODERS
# --------------------------------
def load_model():
    df = pd.read_csv(DATA_PATH)

    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()

    num_languages = df["language_idx"].nunique()

    model = NCF(
        num_users=num_users,
        num_items=num_items,
        num_languages=num_languages
)


    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)

    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)

    with open(LANGUAGE_ENCODER_PATH, "rb") as f:
        language_encoder = pickle.load(f)

    user_language_df = pd.read_csv(USER_LANGUAGE_MAP_PATH)


    return model, user_encoder, item_encoder, language_encoder, user_language_df, num_items

# RECOMMEND FUNCTION
# --------------------------------
def recommend_for_user(user_id, top_k=10):
    (
        model,
        user_encoder,
        item_encoder,
        language_encoder,
        user_language_df,
        num_items
    ) = load_model()

    if user_id not in user_encoder.classes_:
        raise ValueError("User not found in training data")

    user_idx = user_encoder.transform([user_id])[0]

    # Get user's language
    row = user_language_df[user_language_df["user_id"] == user_id]

    if row.empty:
        # 🔁 Fallback: random language
        language_idx = torch.randint(
            low=0,
            high=len(language_encoder.classes_),
            size=(1,)
        ).item()
    else:
        language_id = row.iloc[0]["language_id"]
        language_idx = language_encoder.transform([language_id])[0]

    # Prepare tensors
    item_indices = torch.arange(num_items, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((num_items,), user_idx, dtype=torch.long).to(DEVICE)
    language_tensor = torch.full((num_items,), language_idx, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        scores = model(user_tensor, item_indices, language_tensor)

    top_items = torch.topk(scores, top_k).indices.cpu().numpy()
    recommended_audio_ids = item_encoder.inverse_transform(top_items)

    return recommended_audio_ids

if __name__ == "__main__":
    # Replace with an actual user_id from interactions.csv
    sample_user_id = None  

    df = pd.read_csv("data/processed/interactions.csv")
    sample_user_id = df["user_id"].iloc[0]

    recs = recommend_for_user(sample_user_id, top_k=5)
    print("Recommended audio IDs:")
    for r in recs:
        print(r)
