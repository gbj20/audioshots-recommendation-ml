import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Neural network layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_idx, item_idx):
        # Look up embeddings
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        # Concatenate user & item vectors
        x = torch.cat([user_vec, item_vec], dim=1)

        # Predict interaction score
        score = self.mlp(x)
        return score.squeeze()
