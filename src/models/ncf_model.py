import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering with Content-Based Filtering

    Key improvement: Items have FIXED language and category attributes.
    This ensures the model learns that filtering by language/category 
    actually filters the items, not just adds features.
    """

    def __init__(self, num_users, num_items, num_languages, num_categories, embedding_dim=32):
        super().__init__()

        # User and Item embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Language and Category embeddings (smaller)
        self.language_emb = nn.Embedding(num_languages, embedding_dim // 2)
        self.category_emb = nn.Embedding(num_categories, embedding_dim // 2)

        # Content feature network (combines item with its language/category)
        self.content_fc = nn.Sequential(
            nn.Linear(embedding_dim + (embedding_dim // 2) * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final prediction network
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # user_emb + content_emb
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item, language, category):
        # Get embeddings
        u = self.user_emb(user)  # User preferences
        i = self.item_emb(item)  # Item features
        l = self.language_emb(language)  # Language features
        c = self.category_emb(category)  # Category features

        item_content = torch.cat([i, l, c], dim=1)
        item_content = self.content_fc(item_content)

        # Combine user preferences with content-aware item
        x = torch.cat([u, item_content], dim=1)

        return self.fc(x).squeeze()


class ContentFilteredNCF(nn.Module):
    """
    ALTERNATIVE: More explicit content filtering

    This version adds a filtering layer that explicitly checks
    if the item matches the requested language/category.
    """

    def __init__(self, num_users, num_items, num_languages, num_categories, embedding_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.language_emb = nn.Embedding(num_languages, embedding_dim // 2)
        self.category_emb = nn.Embedding(num_categories, embedding_dim // 2)

        # Store item metadata (will be set during training)
        self.register_buffer('item_languages', torch.zeros(
            num_items, dtype=torch.long))
        self.register_buffer('item_categories', torch.zeros(
            num_items, dtype=torch.long))

        # Compatibility layers (checks if item matches query)
        self.lang_compatibility = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        self.cat_compatibility = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # Main prediction network
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def set_item_metadata(self, item_languages, item_categories):
        """
        Set the language and category for each item.
        Call this after creating the model.

        Args:
            item_languages: tensor of shape [num_items] with language index for each item
            item_categories: tensor of shape [num_items] with category index for each item
        """
        self.item_languages = item_languages
        self.item_categories = item_categories

    def forward(self, user, item, language, category):
        # Get base embeddings
        u = self.user_emb(user)
        i = self.item_emb(item)
        l = self.language_emb(language)
        c = self.category_emb(category)

        # Get actual item metadata
        item_lang = self.language_emb(self.item_languages[item])
        item_cat = self.category_emb(self.item_categories[item])

        lang_match = self.lang_compatibility(torch.abs(l - item_lang))
        cat_match = self.cat_compatibility(torch.abs(c - item_cat))

        # Combine user and item
        x = torch.cat([u, i], dim=1)
        base_score = self.fc(x).squeeze()

        content_filter = (lang_match * cat_match).squeeze()

        return base_score * content_filter
