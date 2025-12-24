import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_languages,
        embedding_dim=32
    ):
        super(NCF, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.language_embedding = nn.Embedding(num_languages, embedding_dim)

        # Neural network
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_idx, item_idx, language_idx):
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        language_vec = self.language_embedding(language_idx)

        x = torch.cat([user_vec, item_vec, language_vec], dim=1)
        score = self.mlp(x)

        return score.squeeze()
