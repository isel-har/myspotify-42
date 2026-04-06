from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset#, DataLoader
import numpy as np
import torch
# import pandas as pd


class UserItemDataset(Dataset):
    def __init__(self, matrix, implicit=False):
        self.samples = []
        
        if not isinstance(matrix, np.ndarray):
            raise Exception("matrix must be a numpy array")

        rows, cols = np.where(matrix > 0)

        for user_id, item_id in zip(rows, cols):
            rating = matrix[user_id, item_id]

            if implicit:
                self.samples.append((user_id, item_id, 1.0))

                neg_item = np.random.choice(np.where(matrix[user_id] == 0)[0])
                self.samples.append((user_id, neg_item, 0.0))
            else:
                self.samples.append((user_id, item_id, float(rating)))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, item, label = self.samples[idx]

        # user_encoder = LabelEncoder()
        # item_encoder = LabelEncoder()

        # user  = user_encoder.fit_transform(user)
        # item  = item_encoder.fit_transform(item)

        return (
            torch.tensor(user,  dtype=torch.long),
            torch.tensor(item,  dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )
