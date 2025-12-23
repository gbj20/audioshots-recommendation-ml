import sys
from pathlib import Path

# --------------------------------
# Fix import path
# --------------------------------
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------
# LOAD MODEL & ENCODERS
# --------------------------------
def load_model():
    df = pd.read_csv(DATA_PATH)

    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()

    model = NCF(num_users, num_items)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)

    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)

    return model, user_encoder, item_encoder, num_items


# --------------------------------
# RECOMMEND FUNCTION
# --------------------------------
def recommend_for_user(user_id, top_k=10):
    model, user_encoder, item_encoder, num_items = load_model()

    # Encode user
    if user_id not in user_encoder.classes_:
        raise ValueError("User not found in training data")

    user_idx = user_encoder.transform([user_id])[0]

    # Create item list
    item_indices = torch.arange(num_items, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((num_items,), user_idx, dtype=torch.long).to(DEVICE)

    # Predict scores
    with torch.no_grad():
        scores = model(user_tensor, item_indices)

    # Get top K items
    top_items = torch.topk(scores, top_k).indices.cpu().numpy()

    # Decode item IDs
    recommended_audio_ids = item_encoder.inverse_transform(top_items)

    return recommended_audio_ids


# --------------------------------
# TEST (TEMPORARY)
# --------------------------------
if __name__ == "__main__":
    # Replace with an actual user_id from interactions.csv
    sample_user_id = None  

    df = pd.read_csv("data/processed/interactions.csv")
    sample_user_id = df["user_id"].iloc[0]

    recs = recommend_for_user(sample_user_id, top_k=5)
    print("Recommended audio IDs:")
    for r in recs:
        print(r)
