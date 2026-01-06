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
EPOCHS = 15  # Increased epochs for better learning
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

print("TRAINING IMPROVED RECOMMENDATION MODEL")
print("="*70 + "\n")

print("Loading training data...")
df = pd.read_csv(DATA_PATH)

num_users = df["user_idx"].nunique()
num_items = df["item_idx"].nunique()
num_languages = df["language_idx"].nunique()
num_categories = df["category_idx"].nunique()

print(f"Users: {num_users}")
print(f"Items: {num_items}")
print(f"Languages: {num_languages}")
print(f"Categories: {num_categories}")
print(f"Total interactions: {len(df)}\n")

dataset = InteractionDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create model
model = NCF(num_users, num_items, num_languages, num_categories, EMBEDDING_DIM)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Learning rate scheduler (reduces LR when loss plateaus)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

print("="*70)
print("TRAINING STARTED")
print("="*70 + "\n")

model.train()
best_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0.0
    batch_count = 0

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
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    
    # Update learning rate based on loss
    scheduler.step(avg_loss)
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        Path("models_saved").mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}  (Best - Model Saved)")
    else:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nBest Loss: {best_loss:.4f}")
print(f"Model saved to: {MODEL_PATH}")

