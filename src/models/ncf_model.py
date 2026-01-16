import torch
import torch.nn as nn


class ImprovedNCF(nn.Module):
    """
    Improved Neural Collaborative Filtering with Content Features
    Uses user, item, language, and category embeddings
    """
    def __init__(self, num_users, num_items, num_languages, num_categories, embedding_dim=64):

        super().__init__()
        
        # Embedding layers
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.language_emb = nn.Embedding(num_languages, embedding_dim // 4)
        self.category_emb = nn.Embedding(num_categories, embedding_dim // 4)
        
        # Calculate total input size
        # user_emb (64) + item_emb (64) + language_emb (16) + category_emb (16) = 160
        input_size = embedding_dim * 2 + (embedding_dim // 4) * 2
        
        # MLP layers with BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.language_emb.weight, std=0.01)
        nn.init.normal_(self.category_emb.weight, std=0.01)
    
    def forward(self, user, item, language, category):
        """
        Forward pass
        
        Args:
            user: User indices [batch_size]
            item: Item indices [batch_size]
            language: Language indices [batch_size]
            category: Category indices [batch_size]
        
        Returns:
            predictions: Predicted scores [batch_size]
        """
        # Get embeddings
        u = self.user_emb(user)       # [batch_size, embedding_dim]
        i = self.item_emb(item)       # [batch_size, embedding_dim]
        l = self.language_emb(language)  # [batch_size, embedding_dim//4]
        c = self.category_emb(category)  # [batch_size, embedding_dim//4]
        
        # Concatenate all features
        x = torch.cat([u, i, l, c], dim=1)  # [batch_size, input_size]
        
        # Pass through MLP
        output = self.fc(x).squeeze()  # [batch_size]
        
        return output


class NCF(nn.Module):
    """
    Original NCF Model (kept for backward compatibility)
    """
    
    def __init__(self, num_users, num_items, num_languages, num_categories, embedding_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.language_emb = nn.Embedding(num_languages, embedding_dim // 2)
        self.category_emb = nn.Embedding(num_categories, embedding_dim // 2)

        self.content_fc = nn.Sequential(
            nn.Linear(embedding_dim + (embedding_dim // 2) * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item, language, category):
        u = self.user_emb(user)
        i = self.item_emb(item)
        l = self.language_emb(language)
        c = self.category_emb(category)

        item_content = torch.cat([i, l, c], dim=1)
        item_content = self.content_fc(item_content)

        x = torch.cat([u, item_content], dim=1)

        return self.fc(x).squeeze()