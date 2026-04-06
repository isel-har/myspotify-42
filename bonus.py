from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dl_recsys import NCFRecommender, NeuMF
from rec_sys import Recommender
import torch

def recommend_top_k(model, user_id, dataset, k=10, device="cpu"):
    model.eval()

    user_tensor = torch.tensor([user_id] * len(dataset)).to(device)
    item_tensor = torch.tensor(dataset).to(device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor)

        # If using BCEWithLogitsLoss → apply sigmoid
        scores = torch.sigmoid(scores)

    scores = scores.cpu().numpy().flatten()

    # Get top-k indices
    top_k_idx = scores.argsort()[-k:][::-1]

    return [(dataset[i], scores[i]) for i in top_k_idx]



def main():
    rec = Recommender()
    triplets = rec.user_item_matrix(only_triples=True, users_limit=50, songs_limit=50)
    users  = triplets['user_id'].to_numpy()
    items  = triplets['song_id'].to_numpy()
    labels = triplets['play_count'].to_numpy()

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    users = user_encoder.fit_transform(users)
    items = item_encoder.fit_transform(items)

    users  = torch.tensor(users,  dtype=torch.long)
    items  = torch.tensor(items,  dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    users_train, users_test, items_train, items_test, y_train, y_test = train_test_split(
        users, items, labels, test_size=0.2, random_state=42
    )

    y_train = (y_train > 0).float().unsqueeze(1)
    y_test  = (y_test > 0).float().unsqueeze(1)


    train_dataset = TensorDataset(users_train, items_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

    num_users = users.max().item() + 1
    num_items = items.max().item() + 1

    nfc = NCFRecommender(num_users=num_users, num_items=num_items, embedding_dim=8)
    # nmf = NeuMF(num_users=num_users, num_items=num_items)

    ## training section
    NCFRecommender.fit(nfc, train_loader, epochs=20)
    # NeuMF.fit(nmf, train_loader, epochs=20)

    prirecommend_top_k(nfc, "123", items_test)

    # criterion = torch.nn.BCEWithLogitsLoss()

    # model.eval()
    # with torch.no_grad():
    #     preds = model(users_test, items_test)
    #     loss  = criterion(preds, y_test)

    # print("Test Loss:", loss.item())
    ### testing section


    print("NFC Approach")





if __name__ == "__main__":
    main()



