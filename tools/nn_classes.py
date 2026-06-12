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



class ContentBasedNN(nn.Module): ## Single user!
    def __init__(
        self,
        num_artists,
        num_genres,
        artist_emb_dim=32,
        genre_emb_dim=8,
        hidden_layers_size=(128, 64)
    ):
        super().__init__()

        self.artist_emb = nn.Embedding(num_artists, artist_emb_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                artist_emb_dim + genre_emb_dim,
                hidden_layers_size[0]
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_layers_size[0],
                hidden_layers_size[1]
                ),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[1], 1),
            nn.Sigmoid()
        )

    def forward(self, artist_id, genre_id):

        artist_vec = self.artist_emb(artist_id)
        genre_vec = self.genre_emb(genre_id)

        x = torch.cat(
            [artist_vec, genre_vec],
            dim=1
        )

        return self.mlp(x)
