import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_languages, num_categories, embedding_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.language_emb = nn.Embedding(num_languages, embedding_dim)
        self.category_emb = nn.Embedding(num_categories, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, item, language, category):
        u = self.user_emb(user)
        i = self.item_emb(item)
        l = self.language_emb(language)
        c = self.category_emb(category)

        x = torch.cat([u, i, l, c], dim=1)
        return self.fc(x).squeeze()
