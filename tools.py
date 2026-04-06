from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd


def compute_item_similarity(train_matrix):
    item_sim = cosine_similarity(train_matrix.T)
    item_sim_df = pd.DataFrame(
        item_sim,
        index=train_matrix.columns,
        columns=train_matrix.columns
    )
    return item_sim_df


def train_test_split_matrix(user_item_matrix, test_size=0.2):
    train = user_item_matrix.copy()
    test  = pd.DataFrame(0, index=user_item_matrix.index, columns=user_item_matrix.columns)

    for user in user_item_matrix.index:
        items = user_item_matrix.loc[user]
        non_zero_items = items[items > 0].index.tolist()

        if len(non_zero_items) < 5:
            continue  # skip users with too few interactions

        n_test = max(1, int(len(non_zero_items) * test_size))
        test_items = np.random.choice(non_zero_items, size=n_test, replace=False)

        # move items from train → test
        train.loc[user, test_items] = 0
        test.loc[user, test_items]  = user_item_matrix.loc[user, test_items]

    return train, test


def recommend(user_id, train_matrix, item_sim_df, k=10):
    if user_id not in train_matrix.index:
        return pd.Series()

    user_vector = train_matrix.loc[user_id]

    scores = item_sim_df.dot(user_vector)

    seen_items = user_vector[user_vector > 0].index

    scores = scores.drop(seen_items, errors='ignore')

    top_k = scores.sort_values(ascending=False).head(k)

    return top_k


def precision_at_k(train, test, item_sim_df, k=10):
    precisions = []

    for user in train.index:
        test_items = test.loc[user]
        test_items = test_items[test_items > 0].index

        if len(test_items) == 0:
            continue

        recs = recommend(user, train, item_sim_df, k)

        hits = len(set(recs.index) & set(test_items))
        precisions.append(hits / k)

    return np.mean(precisions)



def train_svd(train_matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    user_factors = svd.fit_transform(train_matrix)   # U
    item_factors = svd.components_                  # V

    return user_factors, item_factors



def compute_scores(user_factors, item_factors, train_matrix):
    scores = np.dot(user_factors, item_factors)

    return pd.DataFrame(
        scores,
        index=train_matrix.index,
        columns=train_matrix.columns
    )


def recommend_svd(user_id, score_matrix, train_matrix, k=10):
    if user_id not in score_matrix.index:
        return pd.Series()

    user_scores = score_matrix.loc[user_id]

    # remove already listened songs
    seen_items = train_matrix.loc[user_id]
    seen_items = seen_items[seen_items > 0].index

    user_scores = user_scores.drop(seen_items, errors='ignore')
    # top k
    top_k = user_scores.sort_values(ascending=False).head(k)

    return top_k


def precision_at_k_svd(train, test, score_matrix, k=10):
    precisions = []

    for user in train.index:
        test_items = test.loc[user]
        test_items = test_items[test_items > 0].index

        if len(test_items) == 0:
            continue

        recs = recommend_svd(user, score_matrix, train, k)

        hits = len(set(recs.index) & set(test_items))
        precisions.append(hits / k)

    return np.mean(precisions)

