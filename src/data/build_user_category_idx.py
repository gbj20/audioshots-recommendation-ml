import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Paths
USER_CATEGORY_PATH = "data/processed/user_category_map.csv"
INTERACTIONS_PATH = "data/processed/interactions.csv"
USER_ENCODER_PATH = "models_saved/user_encoder.pkl"

OUT_PATH = "data/processed/user_category_idx.csv"
CATEGORY_ENCODER_PATH = "models_saved/category_encoder.pkl"

# Load data
user_cat = pd.read_csv(USER_CATEGORY_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH)

# Load user encoder
with open(USER_ENCODER_PATH, "rb") as f:
    user_encoder = pickle.load(f)

# Keep only users seen in interactions
valid_users = set(interactions["user_id"].unique())
user_cat = user_cat[user_cat["user_id"].isin(valid_users)]

# Encode users
user_cat["user_idx"] = user_encoder.transform(user_cat["user_id"])

# Encode categories
cat_encoder = LabelEncoder()
user_cat["category_idx"] = cat_encoder.fit_transform(user_cat["category"])

# Save
Path("data/processed").mkdir(exist_ok=True)
user_cat[["user_idx", "category_idx"]].to_csv(OUT_PATH, index=False)

Path("models_saved").mkdir(exist_ok=True)
with open(CATEGORY_ENCODER_PATH, "wb") as f:
    pickle.dump(cat_encoder, f)

print("user_category_idx.csv created")
print("Users:", user_cat["user_idx"].nunique())
print("Categories:", user_cat["category_idx"].nunique())
print(user_cat.head())
