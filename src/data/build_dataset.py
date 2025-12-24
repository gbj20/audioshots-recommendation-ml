import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle


df = pd.read_csv("data/processed/interactions.csv")
user_lang = pd.read_csv("data/processed/user_language_map.csv")
df = df.merge(user_lang, on="user_id", how="left")


user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
language_encoder = LabelEncoder()

df["language_idx"] = language_encoder.fit_transform(df["language_id"])
df["user_idx"] = user_encoder.fit_transform(df["user_id"])
df["item_idx"] = item_encoder.fit_transform(df["audio_id"])

ml_df = df[["user_idx", "item_idx", "language_idx", "score"]]


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

with open("models_saved/language_encoder.pkl", "wb") as f:
    pickle.dump(language_encoder, f)


print("✅ ML dataset created")
print("Users:", ml_df["user_idx"].nunique())
print("Items:", ml_df["item_idx"].nunique())
print(ml_df.head())
