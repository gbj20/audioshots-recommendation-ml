import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.ncf_model import NCF

DATA_PATH = "data/processed/ml_interactions.csv"
MODEL_PATH = "models_saved/recommender.pt"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.languages = torch.tensor(df["language_idx"].values, dtype=torch.long)
        self.categories = torch.tensor(df["category_idx"].values, dtype=torch.long)
        self.scores = torch.tensor(df["score"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.languages[idx],
            self.categories[idx],
            self.scores[idx],
        )


# --------------------------------
# LOAD DATA
# --------------------------------
print("Loading training data...")
df = pd.read_csv(DATA_PATH)

num_users = df["user_idx"].nunique()
num_items = df["item_idx"].nunique()

print(f"Users: {num_users}, Items: {num_items}")

dataset = InteractionDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_languages = df["language_idx"].nunique()
num_categories = df["category_idx"].nunique()
model = NCF(num_users, num_items, num_languages, num_categories, EMBEDDING_DIM)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training started...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for user, item, language, category, score in dataloader:
        user = user.to(DEVICE)
        item = item.to(DEVICE)
        language = language.to(DEVICE)
        category = category.to(DEVICE)
        score = score.to(DEVICE)

        optimizer.zero_grad()
        prediction = model(user, item, language, category)
        loss = criterion(prediction, score)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")


Path("models_saved").mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model training complete. Saved at {MODEL_PATH}")
