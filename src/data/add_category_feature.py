import pandas as pd
import pickle
from pathlib import Path

# -----------------------------
# Load datasets
# -----------------------------
ml_df = pd.read_csv("data/processed/ml_interactions.csv")
user_category_df = pd.read_csv("data/processed/user_category_map.csv")

print("ML interactions columns:", ml_df.columns.tolist())
print("User-category columns:", user_category_df.columns.tolist())

# -----------------------------
# Load user encoder (idx -> user_id)
# -----------------------------
with open("models_saved/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

# Reverse mapping: idx -> user_id
idx_to_user = {idx: user_id for idx, user_id in enumerate(user_encoder.classes_)}

# Add real user_id column
ml_df["user_id"] = ml_df["user_idx"].map(idx_to_user)

# -----------------------------
# Merge category info
# -----------------------------
merged = ml_df.merge(
    user_category_df,
    on="user_id",
    how="left"
)

# -----------------------------
# Encode categories
# -----------------------------
merged["category"] = merged["category"].fillna("UNKNOWN")

category_encoder = {cat: idx for idx, cat in enumerate(merged["category"].unique())}
merged["category_idx"] = merged["category"].map(category_encoder)

# -----------------------------
# Save encoders
# -----------------------------
Path("models_saved").mkdir(exist_ok=True)
with open("models_saved/category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)

# -----------------------------
# Final ML dataset
# -----------------------------
final_df = merged[[
    "user_idx",
    "item_idx",
    "language_idx",
    "category_idx",
    "score"
]]

final_df.to_csv("data/processed/ml_interactions.csv", index=False)

print("✅ Category feature added successfully")
print(final_df.head())
