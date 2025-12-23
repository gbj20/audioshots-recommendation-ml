import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle

# -----------------------------
# Load interaction data
# -----------------------------
df = pd.read_csv("data/processed/interactions.csv")

# -----------------------------
# Encode userId and audioId
# -----------------------------
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["user_idx"] = user_encoder.fit_transform(df["user_id"])
df["item_idx"] = item_encoder.fit_transform(df["audio_id"])


# -----------------------------
# Keep only required columns
# -----------------------------
ml_df = df[["user_idx", "item_idx", "score"]]

# -----------------------------
# Save processed dataset
# -----------------------------
Path("data/processed").mkdir(exist_ok=True)
ml_df.to_csv("data/processed/ml_interactions.csv", index=False)

# -----------------------------
# Save encoders (VERY IMPORTANT)
# -----------------------------
Path("models_saved").mkdir(exist_ok=True)

with open("models_saved/user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)

with open("models_saved/item_encoder.pkl", "wb") as f:
    pickle.dump(item_encoder, f)

print("✅ ML dataset created")
print("Users:", ml_df["user_idx"].nunique())
print("Items:", ml_df["item_idx"].nunique())
print(ml_df.head())
