import pandas as pd
from pathlib import Path

# Load users data
users = pd.read_csv("data/raw/AudioShots_UAT.users.csv")

# Identify category columns
category_cols = [c for c in users.columns if c.startswith("categories[")]

rows = []

for _, row in users.iterrows():
    user_id = row["_id"]
    for col in category_cols:
        cat = row[col]
        if pd.notna(cat):
            rows.append({
                "user_id": user_id,
                "category": cat.strip()
            })

user_category_df = pd.DataFrame(rows)

# Save
Path("data/processed").mkdir(exist_ok=True)
user_category_df.to_csv("data/processed/user_category_map.csv", index=False)

print("user_category_map.csv created")
print(user_category_df.head())
print("Unique categories:", user_category_df["category"].nunique())
