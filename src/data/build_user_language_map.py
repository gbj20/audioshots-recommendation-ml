import pandas as pd
from pathlib import Path

# -----------------------------
# Load users data
# -----------------------------
users = pd.read_csv("data/raw/AudioShots_UAT.users.csv")

# -----------------------------
# Extract primary language
# -----------------------------
language_cols = ["languages[0]", "languages[1]", "languages[2]"]

def get_primary_language(row):
    for col in language_cols:
        if pd.notna(row[col]):
            return row[col]
    return None

users["language_id"] = users.apply(get_primary_language, axis=1)

# -----------------------------
# Build user-language mapping
# -----------------------------
user_language_map = users[["_id", "language_id"]].dropna()
user_language_map.columns = ["user_id", "language_id"]

# -----------------------------
# Save
# -----------------------------
Path("data/processed").mkdir(parents=True, exist_ok=True)
user_language_map.to_csv("data/processed/user_language_map.csv", index=False)

print("✅ user_language_map.csv created")
print(user_language_map.head())
print("Total users with language:", len(user_language_map))
