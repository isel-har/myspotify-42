import torch.nn as nn
import torch

class NCFRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8):
        super(NCFRecommender, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(x)


class ItemEncoder(nn.Module):
    def __init__(self, n_artists, n_genres, d=64):
        super().__init__()

        self.artist_emb = nn.Embedding(n_artists, d)
        self.genre_emb  = nn.Embedding(n_genres, d)

        self.mlp = nn.Sequential(
            nn.Linear(d * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, artist_id, genre_id):
        x = torch.cat([
            self.artist_emb(artist_id),
            self.genre_emb(genre_id)
        ], dim=1)
        return self.mlp(x)


class UserEncoder(nn.Module):
    def __init__(self, n_users, d=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)

    def forward(self, user_id):
        return self.user_emb(user_id)

