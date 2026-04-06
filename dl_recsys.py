import torch
import torch.nn as nn



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
    
    @staticmethod
    def fit(model, train_loader, epochs=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for users, items, labels in train_loader:
                optimizer.zero_grad()

                predictions = model(users, items)
                loss        = criterion(predictions, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        return model



class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=16, mlp_dim=32):
        super().__init__()
        # MF embeddings
        self.mf_user = nn.Embedding(num_users, mf_dim)
        self.mf_item = nn.Embedding(num_items, mf_dim)
        # MLP embeddings
        self.mlp_user = nn.Embedding(num_users, mlp_dim)
        self.mlp_item = nn.Embedding(num_items, mlp_dim)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 32),          nn.ReLU(),
        )
        self.output = nn.Linear(mf_dim + 32, 1)

    def forward(self, user, item):
        # MF path: element-wise product
        mf_out = self.mf_user(user) * self.mf_item(item)

        # MLP path: concatenation through layers
        mlp_in = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=1)
        mlp_out = self.mlp(mlp_in)

        # Merge both paths
        combined = torch.cat([mf_out, mlp_out], dim=1)
        return self.output(combined)