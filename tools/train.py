import torch

def train(model, train_loader, epochs=10):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for users, items, labels in train_loader:
            optimizer.zero_grad()

            predictions = model(users, items)
            loss        = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        return model
