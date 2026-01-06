from src.models.ncf_model import NCF
import pickle
import pandas as pd
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models_saved/recommender.pt"
DATA_PATH = PROJECT_ROOT / "data/processed/ml_interactions.csv"

USER_ENCODER_PATH = PROJECT_ROOT / "models_saved/user_encoder.pkl"
ITEM_ENCODER_PATH = PROJECT_ROOT / "models_saved/item_encoder.pkl"
LANGUAGE_ENCODER_PATH = PROJECT_ROOT / "models_saved/language_encoder.pkl"
CATEGORY_ENCODER_PATH = PROJECT_ROOT / "models_saved/category_encoder.pkl"

USER_LANGUAGE_MAP_PATH = PROJECT_ROOT / "data/processed/user_language_map.csv"
USER_CATEGORY_MAP_PATH = PROJECT_ROOT / "data/processed/user_category_idx.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    df = pd.read_csv(DATA_PATH)

    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()
    num_languages = df["language_idx"].nunique()
    num_categories = df["category_idx"].nunique()

    model = NCF(
        num_users=num_users,
        num_items=num_items,
        num_languages=num_languages,
        num_categories=num_categories
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

    with open(CATEGORY_ENCODER_PATH, "rb") as f:
        category_encoder = pickle.load(f)

    user_language_df = pd.read_csv(USER_LANGUAGE_MAP_PATH)
    user_category_df = pd.read_csv(USER_CATEGORY_MAP_PATH)

    return (
        model,
        user_encoder,
        item_encoder,
        language_encoder,
        category_encoder,
        user_language_df,
        user_category_df,
        num_items
    )
def recommend_for_user(user_id, top_k=10):
    (
        model,
        user_encoder,
        item_encoder,
        language_encoder,
        category_encoder,
        user_language_df,
        user_category_df,
        num_items
    ) = load_model()

    # Convert user_id → user_idx
    if user_id not in user_encoder.classes_:
        raise ValueError("User not found")

    user_idx = user_encoder.transform([user_id])[0]

    # Language index
    lang_row = user_language_df[user_language_df["user_id"] == user_id]

    if lang_row.empty:
        language_idx = torch.randint(
            0, len(language_encoder.classes_), (1,)).item()
    else:
        language_id = lang_row.iloc[0]["language_id"]
        language_idx = language_encoder.transform([language_id])[0]

    # Category index 
    cat_row = user_category_df[user_category_df["user_idx"] == user_idx]

    if cat_row.empty:
        category_idx = torch.randint(
            0, len(category_encoder.classes_), (1,)).item()
    else:
        category_idx = cat_row.iloc[0]["category_idx"]

    # Tensors
    item_tensor = torch.arange(num_items, dtype=torch.long).to(DEVICE)
    user_tensor = torch.full((num_items,), user_idx,
                             dtype=torch.long).to(DEVICE)
    language_tensor = torch.full(
        (num_items,), language_idx, dtype=torch.long).to(DEVICE)
    category_tensor = torch.full(
        (num_items,), category_idx, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor,
                       language_tensor, category_tensor)

    top_items = torch.topk(scores, top_k).indices.cpu().numpy()
    return item_encoder.inverse_transform(top_items)
