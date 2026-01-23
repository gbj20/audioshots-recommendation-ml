from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.ncf_model import ImprovedNCF

DATA_PATH = "data/processed/ml_interactions.csv"
MODEL_PATH = "models_saved/recommender_improved.pt"

EPOCHS = 50 
LEARNING_RATE = 0.0001  
EMBEDDING_DIM = 64 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.languages = torch.tensor(
            df["language_idx"].values, dtype=torch.long)
        self.categories = torch.tensor(
            df["category_idx"].values, dtype=torch.long)
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

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_mse = 0
    total_mae = 0
    count = 0

    predictions_list = []
    actuals_list = []

    with torch.no_grad():
        for user, item, language, category, score in dataloader:
            user = user.to(device)
            item = item.to(device)
            language = language.to(device)
            category = category.to(device)
            score = score.to(device)

            prediction = model(user, item, language, category)

            mse = criterion(prediction, score)
            mae = torch.abs(prediction - score).mean()

            total_mse += mse.item()
            total_mae += mae.item()
            count += 1

            predictions_list.extend(prediction.cpu().numpy())
            actuals_list.extend(score.cpu().numpy())

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    rmse = np.sqrt(avg_mse)

    # Check prediction variance
    pred_std = np.std(predictions_list)

    return avg_mse, avg_mae, rmse, pred_std


def train_model():
    print("\n" + "="*80)
    print("TRAINING MODEL - OPTIMIZED FOR SMALL/SPARSE DATA")
    print("="*80 + "\n")

    # Load data
    df = pd.read_csv(DATA_PATH)

    num_users = df["user_idx"].max() + 1
    num_items = df["item_idx"].max() + 1
    num_languages = df["language_idx"].max() + 1
    num_categories = df["category_idx"].max() + 1

    print(f"Dataset Statistics:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Languages: {num_languages}")
    print(f"  Categories: {num_categories}")
    print(f"  Total interactions: {len(df)}")

    # Check data distribution
    print(f"\nScore Distribution:")
    for score, count in df['score'].value_counts().sort_index().items():
        print(f"  Score {score}: {count} ({count/len(df)*100:.1f}%)")

    sparsity = 1 - len(df) / (num_users * num_items)
    print(f"\nData sparsity: {sparsity:.2%}")

    if sparsity > 0.95:
        print("   WARNING: Very sparse data! Model may struggle.")
    print()

    # Train/Test split
    try:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['user_idx']
        )
    except ValueError:
        # Users with only 1 interaction can't be stratified
        print("   Some users have only 1 interaction - using simple split\n")
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )

    # Create dataloaders
    train_dataset = InteractionDataset(train_df)
    test_dataset = InteractionDataset(test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    print(f"Initializing model on: {DEVICE}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Learning rate: {LEARNING_RATE}\n")

    model = ImprovedNCF(num_users, num_items, num_languages,
                         num_categories, embedding_dim=EMBEDDING_DIM)
    model = model.to(DEVICE)

    # Optimizer with strong weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Cosine annealing scheduler (better for small data)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    print("="*80)
    print("TRAINING STARTED")
    print("="*80 + "\n")

    best_test_rmse = float('inf')
    best_pred_std = 0
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0

        for user, item, language, category, score in train_loader:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            language = language.to(DEVICE)
            category = category.to(DEVICE)
            score = score.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(user, item, language, category)
            loss = criterion(prediction, score)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation
        test_mse, test_mae, test_rmse, pred_std = evaluate_model(
            model, test_loader, DEVICE, criterion)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test RMSE:  {test_rmse:.4f} | MAE: {test_mae:.4f}")
            print(
                f"  Prediction Std: {pred_std:.4f} (higher = more diverse predictions)")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Check if model is collapsed
            if pred_std < 0.1:
                print(f"  ⚠ WARNING: Model predictions have low variance (collapsed)")

            if test_rmse < best_test_rmse and pred_std > 0.1:
                best_test_rmse = test_rmse
                best_pred_std = pred_std
                patience_counter = 0

                Path("models_saved").mkdir(exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"   Best model saved!")
            else:
                patience_counter += 1

            print()

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))
    test_mse, test_mae, test_rmse, pred_std = evaluate_model(
        model, test_loader, DEVICE, criterion)

    print("FINAL TEST METRICS:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  Prediction Std: {pred_std:.4f}")
    print()

    # Sample predictions to verify
    print("SAMPLE PREDICTIONS (verifying model works):")
    print("-"*80)
    sample = test_df.sample(min(5, len(test_df)))

    model.eval()
    with torch.no_grad():
        for _, row in sample.iterrows():
            user_t = torch.tensor(
                [row['user_idx']], dtype=torch.long).to(DEVICE)
            item_t = torch.tensor(
                [row['item_idx']], dtype=torch.long).to(DEVICE)
            lang_t = torch.tensor([row['language_idx']],
                                  dtype=torch.long).to(DEVICE)
            cat_t = torch.tensor([row['category_idx']],
                                 dtype=torch.long).to(DEVICE)

            pred = model(user_t, item_t, lang_t, cat_t).item()
            print(
                f"True: {row['score']:.1f} | Predicted: {pred:.2f} | Error: {abs(row['score']-pred):.2f}")

    print()

    if test_mae < 1.0 and pred_std > 0.3:
        print(" Model quality: GOOD")
    elif test_mae < 1.5 and pred_std > 0.2:
        print(" Model quality: MODERATE - Consider getting more data")
    else:
        print(" Model quality: POOR - Dataset too small/sparse")
        print("\nSuggestions:")
        print("  1. Collect more user interactions")
        print("  2. Try using only users with 5+ interactions")
        print("  3. Consider a simpler popularity-based baseline")

    print(f"\nModel saved to: {MODEL_PATH}\n")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_model()