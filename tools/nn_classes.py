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



class SongEncoder(nn.Module):
    def __init__(self, num_artists, num_genres, emb_dim=64):
        super().__init__()

        self.artist_emb = nn.Embedding(num_artists, 32)
        self.genre_emb  = nn.Embedding(num_genres, 16)

        self.encoder = nn.Sequential(
            nn.Linear(48, emb_dim),
            nn.ReLU()
        )

        self.artist_head = nn.Linear(emb_dim, num_artists)
        self.genre_head  = nn.Linear(emb_dim, num_genres)

    def forward(self, artist, genre):
        a = self.artist_emb(artist)
        g = self.genre_emb(genre)

        x = torch.cat([a, g], dim=1)
        z = self.encoder(x)

        artist_logits = self.artist_head(z)
        genre_logits  = self.genre_head(z)

        return z, artist_logits, genre_logits


# class UserTasteAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, item_embeddings):  # (N, embed_dim)
#         scores = self.attention(item_embeddings)         # (N, 1)
#         weights = torch.softmax(scores, dim=0)           # (N, 1)
#         taste_vec = (weights * item_embeddings).sum(0)   # (embed_dim,)
#         return taste_vec, weights