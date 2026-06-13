import os

os.environ['GENSIM_DATA_DIR'] = '/Users/isel-har/goinfre/gensim'

import numpy as np
import pandas as pd
import duckdb
import joblib
import torch
import nltk
import torch.nn as nn

import gensim.downloader as api
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tools.nn_classes import NCFRecommender, SongEncoder
from tools.train import train
from torch.utils.data import DataLoader, TensorDataset, Subset
from tools.tools import  compute_item_similarity, recommend, precision_at_k
from tools.tools import train_svd, recommend_svd, compute_scores, precision_at_k_svd
from sklearn.metrics.pairwise import cosine_similarity
import faiss


pd.set_option('display.max_rows', None)

class Recommender:

    stop_words = set(stopwords.words("english")) 

    def __init__(self):

        self.collection_pass = False
        self.is_loaded = False
        self.clf = None
        self.classes_ = None
        self.track_ids = None
        self.y_pred = None
        self.pred_db = None
        self.w2v_model = None
        self.processed = False

        self.themes = {
            'love': ['love', 'heart', 'kiss', 'romance', 'lover', 'baby', 'honey'],
            'war': ['war', 'fight', 'battle', 'soldier', 'gun', 'blood'],
            'happiness': ['happy', 'joy', 'smile', 'laugh', 'celebration', 'party'],
            'loneliness': ['lonely', 'alone', 'sad', 'cry', 'empty', 'miss'],
            'money': ['money', 'rich', 'dollar', 'gold', 'cash', 'wealth']
        }

        self.mds_tagtraum_db = """read_csv('data/p02_msd_tagtraum_cd2.cls', comment='#', columns={
                    'track_id':'VARCHAR',
                    'genre':'VARCHAR'
                },
                ignore_errors=true,
                delim='\t'
        )"""

        self.train_triplets_db = """read_csv('data/train_triplets.txt', delim='\t', header = false,
                    columns={'user_id':'VARCHAR', 'song_id':'VARCHAR', 'play_count':'INTEGER'}
        )"""

        self.unique_tracks = """read_csv('data/p02_unique_tracks.txt',
                                    delim='\n',
                                    header = false,
                                    columns={'line':'VARCHAR'}
                            )"""

        self.unique_tracks_db = f"""SELECT string_split(line, '<SEP>') AS parts
                         FROM {self.unique_tracks}"""




    def collection_filter_query(self, theme, db):
        return f"""
            SELECT 
                tt.artist,
                tt.title,
                SUM(tt.play_count) AS play_count
            FROM
                ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')}) AS tt
            JOIN (select * from {db} where theme like '{theme}') as tm
                ON tt.track_id = tm.track_id
            GROUP BY    
                tt.track_id, tt.artist, tt.title

            ORDER BY play_count DESC
            LIMIT 50
        """


    def triplets_tracks_db(self, columns):
        return f"""
            SELECT
            {columns}
            FROM
            {self.train_triplets_db} as sdb
            JOIN
            (
                    SELECT
                         parts[1]::VARCHAR AS track_id,
                         parts[2]::VARCHAR AS song_id,
                         parts[3]::VARCHAR AS artist,
                         parts[4]::VARCHAR AS title
                     FROM ({self.unique_tracks_db})
            )  AS tdb
            ON sdb.song_id=tdb.song_id
        """


    def top_250_tracks(self):

        result = duckdb.sql(f"""
            {self.triplets_tracks_db("tdb.artist, tdb.title, SUM(sdb.play_count) as play_count")}
            GROUP BY tdb.track_id, tdb.title, tdb.artist
            ORDER BY play_count DESC
            LIMIT 250
        """)
        return result


    def get_genres(self):

        return duckdb.query(f"""
            select distinct genre
            from {self.mds_tagtraum_db}
        """).to_df()['genre'].tolist()


    def top_100_tracks_by_genre(self, genre):
    
        cte = f"""
            WITH tracks_table AS (
                SELECT 
                    tdb.track_id,
                    tdb.artist,
                    tdb.title,
                    SUM(sdb.play_count) AS play_count
                FROM {self.train_triplets_db} AS sdb
                JOIN (
                    SELECT
                        parts[1]::VARCHAR AS track_id,
                        parts[2]::VARCHAR AS song_id,
                        parts[3]::VARCHAR AS artist,
                        parts[4]::VARCHAR AS title
                    FROM ({self.unique_tracks_db})
                ) AS tdb
                ON sdb.song_id = tdb.song_id
                GROUP BY tdb.track_id, tdb.artist, tdb.title
            )
        """

        base_query = f"""
            {cte}
            SELECT tt.artist, tt.title, tt.play_count
            FROM tracks_table AS tt
            JOIN {self.mds_tagtraum_db} AS mds
            ON tt.track_id = mds.track_id
        """

        return duckdb.query(f"""
                {base_query}
                WHERE mds.genre like '{genre}'
                ORDER BY tt.play_count DESC
                LIMIT 100
            """)



    def word_vec(self, theme,top_n=10):
    
        if theme in self.w2v_model:
            return self.w2v_model.most_similar(theme, topn=top_n)
        return None


    def collection(self, theme, threshold=0.1, word2vec=False, top_n=10, min_theme_words=5):

        collection = {'track_id': [], 'theme':[], 'theme_ratio':[]}

        theme_index = set()

        for val in self.themes.get(theme, []):
            idx = self.keyword_map.get(val)
            if idx is not None:
                theme_index.add(idx)
            
        if word2vec:
            if self.w2v_model is None:
                self.w2v_model = api.load("word2vec-google-news-300")

            similar_tokens = self.word_vec(theme=theme, top_n=top_n)

            for token, score in similar_tokens:
                idx = self.keyword_map.get(token)
                if idx is not None:
                    theme_index.add(idx)
        
        with open('data/mxm_dataset_train.txt', 'r') as f:

            for line in f:

                if line.startswith(('%', '#')):
                    continue

                parts = line.strip().split(',')
                track_id = parts[0]

                theme_score = 0
                total_words = 0

                for part in parts[2:]:

                    word_index, count = part.split(':')
                    word_index        = int(word_index)
                    count             = int(count)

                    total_words += count

                    if word_index in theme_index:
                        theme_score += count

                if total_words == 0:
                    continue
                
                theme_ratio = theme_score / total_words

                if theme_ratio > threshold and theme_score >= min_theme_words:
                        collection['track_id'].append(track_id)
                        collection['theme_ratio'].append(theme_ratio)


        collection['theme'] = theme
        data_frame = pd.DataFrame(collection)

        if data_frame.empty:
            print("No tracks found for theme:", theme)
            return pd.DataFrame()

        theme_db = duckdb.from_df(data_frame)
        result   = duckdb.query(f"""
            SELECT 
                tt.artist,
                tt.title,
                SUM(tt.play_count) AS play_count
            FROM
                ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')}) AS tt
            JOIN theme_db tm
                ON tt.track_id = tm.track_id
            GROUP BY 
                tt.track_id, tt.artist, tt.title
            ORDER BY play_count DESC
            LIMIT 50
        """)

        print(data_frame.shape)
        if word2vec:
            data_frame.to_csv(f'data/{theme}_data.csv', index=False, header=False)
            print(f"data frame saved as data/{theme}_data.csv for training")
        return result


    def mxm_dict(self):

        m_dict = {}

        with open('data/mxm_dataset_train.txt', 'r') as f:
            
            for line in f:

                words_count =  np.zeros(self.vocabulary_size)       
                if line.startswith(('%', '#')):
                    continue

                parts    = line.strip().split(',')
                track_id = parts[0]
                for part in parts[2:]:

                    word_index, count     = part.split(':')
                    word_index            = int(word_index)
                    count                 = int(count)
                    if word_index not in self.stop_words_idx:
                        words_count[word_index - 1] = count
                
                m_dict[track_id] = words_count

        return m_dict


    def vectorizer(self, mxm_dict=None, track_id_list=None):
        
        vectors_list = []
        for track_id in track_id_list:
            try:
                vectors_list.append(mxm_dict[track_id])
            except Exception as e:
                print("track id :", track_id)
                print("error :", str(e))
        return np.array(vectors_list)  



    def preprocessing(self):
        
        df = None
        for theme in ['love', 'war', 'happiness']:
            theme_df = pd.read_csv(
                f"data/{theme}_data.csv",
                names=['track_id', 'theme', 'theme_ratio'],
                header=None
            )
            df = pd.concat([df, theme_df])

        df = df.sample(frac=1)
        df = df.drop_duplicates()

        X_train, X_test, y_train, y_test = train_test_split(
            df['track_id'],
            df['theme'],
            test_size=0.2,
            stratify=df['theme'],
            random_state=42
        )

        le   = LabelEncoder()
        mx_dict = self.mxm_dict()

        joblib.dump(X_test, "data/track_ids_test.pkl")
        # joblib.dump(X_train, "data/track_ids_train.pkl")
        # joblib.dump(y_train, "data/track_themes.pkl")

        X_train = self.vectorizer(mx_dict, X_train.values.tolist()).astype(np.float32)
        X_test  = self.vectorizer(mx_dict, X_test.values.tolist()).astype(np.float32)


        le.fit(pd.concat([y_train, y_test]))
        y_train  = le.transform(y_train)
        y_test  =  le.transform(y_test)
    
        joblib.dump(X_train, "data/X_train.pkl")
        joblib.dump(X_test, "data/X_test.pkl")
        joblib.dump(y_train, "data/y_train.pkl")
        joblib.dump(y_test, "data/y_test.pkl")
        joblib.dump(le.classes_, "data/classes.pkl")

        print("preprocessed train/test split and classes saved at data/")


    def classifier(self):
        
        X_train, y_train, X_test, y_test = joblib.load('data/X_train.pkl'), joblib.load('data/y_train.pkl'), \
            joblib.load('data/X_test.pkl'), joblib.load('data/y_test.pkl')
    
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            batch_size=16,
            random_state=42,
            max_iter=100,
            solver='adam',
            activation='relu',
            early_stopping=True
        )

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        self.y_pred = clf.predict(X_test)
        print(f"classifier accuracy reached on test set: {accuracy_score(y_pred=self.y_pred, y_true=y_test)}")
        return clf


    def collection_classification(self, theme):
    
        if not self.is_loaded:
            self.classes_, self.track_ids = joblib.load('data/classes.pkl'), joblib.load('data/track_ids_test.pkl')
            self.clf = self.classifier()

            themes = [self.classes_[l] for l in self.y_pred.tolist()]
            self.pred_db = duckdb.from_df(
                pd.DataFrame({
                    'track_id':self.track_ids,
                    'theme': themes
                })
            )

            self.is_loaded = True

        pred_db = self.pred_db
        query   = self.collection_filter_query(theme, "pred_db")
        result = duckdb.query(query)
        return result
        # rows = len(result)
        # if rows < 50:
        #     original_df = pd.read_csv(f"data/{theme}_data.csv")
        #     query_df = result.to_df()
        #     merged = original_df.merge(query_df, how='left', indicator=True, on='track_id')
        #     not_in_query = duckdb.from_df(merged[merged['_merge'] == 'left_only'].drop(columns=['_merge']))
        #     query = self.collection_filter_query(theme, "not_in_query")
        #     result = duckdb.query(query)

        # return result



        
    def collections(self, theme, process=False, approach='baseline'):
        
        if not self.collection_pass:
            keywords = pd.read_csv('data/mxm_dataset_train.txt', comment='#', nrows=1) \
                .columns.to_list()
            keywords[0] = 'i'
            self.keyword_map = {
                w: i for i, w in enumerate(keywords, start=1)
                if w not in self.stop_words 
            }
            self.vocabulary_size = max( self.keyword_map.values() )
            self.stop_words_idx = {
                w: i for i, w in enumerate(keywords, start=1)
                if w in self.stop_words
            }

            self.collection_pass = True

        print(f"theme : {theme}")
        if approach in ('baseline', 'word2vec'):

            is_wv = approach == 'word2vec'
            return self.collection(
                theme,
                threshold=0.063,
                word2vec=is_wv,
                top_n=10,
                min_theme_words=4
            )
        
        self.w2v_model = None

        if process:
            if not self.processed:
                print("data processing...")
                self.preprocessing()
                self.processed = True
        
        return self.collection_classification(theme)



    def cosine_similarity_approach(self, user_id, train_matrix, test_matrix):
        
        items_sim = compute_item_similarity(train_matrix)

        p_at_10 = precision_at_k(train_matrix, test_matrix, items_sim, k=10)

        top_10_rec = recommend(
            user_id=user_id,
            train_matrix=train_matrix,
            item_sim_df=items_sim,
            k=10
        )
        return top_10_rec, p_at_10



    def matrix_factorization_approach(self, user_id, train_matrix, test_matrix):## SVD
        
        user_factors, item_factors = train_svd(train_matrix, n_components=50)

        score_matrix = compute_scores(user_factors, item_factors, train_matrix)

        p_at_10      = precision_at_k_svd(train_matrix, test_matrix, score_matrix)

        top_10_rec   = recommend_svd(user_id, test_matrix, train_matrix)

        return top_10_rec, p_at_10


    def user_item_matrix(self, only_triples=False, songs_limit=80, users_limit=50):
        popular_songs = f"""
            (select song_id, count(user_id) as shared_users_count from {self.train_triplets_db}
            group by song_id
            order by shared_users_count desc
            limit {songs_limit})
        """
    
        users_listen_popular_songs = f"""
            (
                select user_id, count(song_id) as shared_song_count from {self.train_triplets_db}
                where song_id in (select song_id from {popular_songs})
                group by user_id
                order by shared_song_count desc
                limit {users_limit}
            )
        """

        selected_rows = f"""
            select tt.user_id, tt.song_id, tt.play_count from {self.train_triplets_db} as tt
            join (select user_id from {users_listen_popular_songs}) as up on up.user_id = tt.user_id
            where tt.song_id in (select song_id from {popular_songs})
        """

        db = duckdb.query(selected_rows)
        if only_triples:
            return db.df()

        user_item_matrix_ = db.df().pivot_table(
            index='user_id',
            columns='song_id',
            values='play_count',
            fill_value=0
        )
        return user_item_matrix_



    def user_based_recommendation(self, user_id, train_matrix, test_matrix, baseline=True):
        
        if baseline:
            return self.cosine_similarity_approach(user_id, train_matrix, test_matrix)

        return self.matrix_factorization_approach(user_id, train_matrix, test_matrix)



    def user_item_tensor(self, users_limit=50, songs_limit=50):

        triplets = self.user_item_matrix(only_triples=True, users_limit=users_limit, songs_limit=songs_limit)    
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
        return users, items, labels


    def recommend_top_k_inference(model, user_id, dataset, k=10, device="cpu"):
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
    

    def dataloader_train_test_split(self, users_limit=50, songs_limit=50):
        users, items, labels = self.user_item_tensor(users_limit, songs_limit)
    
        users_train, users_test, items_train, items_test, y_train, y_test = train_test_split(
            users, items, labels, test_size=0.2, random_state=42
        )

        y_train = (y_train > 0).float().unsqueeze(1)
        y_test  = (y_test > 0).float().unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(users_train, items_train, y_train),
            batch_size=32
        )

        test_loader = DataLoader(
            TensorDataset(users_test, items_test, y_test),
            batch_size=32
        )

        return train_loader, test_loader



    # def train(self, model)


    def ncf_recommendation(self, user_id, train_loader, test_loader):

        
        ncf = NCFRecommender(num_users=num_users, num_items=num_items, embedding_dim=8)

        ncf = self.train(ncf, train_loader)

        return self.recommend_top_k_inference(ncf, user_id, test_loader)



    def user_profile_df(self, user_id=None):
        user_history_query = f"""
        with tracks_db as (
            select tdb.track_id, tdb.artist, sdb.play_count from {self.train_triplets_db} as sdb
            join (
                select 
                    parts[1]::VARCHAR AS track_id,
                    parts[2]::VARCHAR AS song_id,
                    parts[3]::VARCHAR AS artist
                    from ({self.unique_tracks_db})
            ) as tdb
            on sdb.song_id = tdb.song_id
            where sdb.user_id like '{user_id}'
            order by sdb.play_count desc
            )
            select tracks_db.*, mdb.genre from tracks_db
            join (
                select * from {self.mds_tagtraum_db}
            ) as mdb
            on mdb.track_id = tracks_db.track_id
        """
        return duckdb.query(user_history_query).to_df()


    # def items_matrix_split(self, user_profile_df):
    #     artist_encoder = OneHotEncoder(sparse_output=False)
    #     genre_encoder  = OneHotEncoder(sparse_output=False)

    #     encoded_artist = artist_encoder.fit_transform(user_profile_df['artist'].to_numpy().reshape(-1, 1))
    #     encoded_genre  = genre_encoder.fit_transform(user_profile_df['genre'].to_numpy().reshape(-1, 1))
    
    #     items_matrix = np.concatenate([encoded_artist, encoded_genre], axis=1)

    #     df_indices = [i for i in range(len(user_profile_df))]

    #     items_train, items_test, indices_train, indices_test = train_test_split(
    #         items_matrix,
    #         df_indices,
    #         test_size=0.2,
    #         shuffle=True,
    #         random_state=42
    #     )
    #     return items_train, items_test, indices_train, indices_test



    def songs_embedding(self):
               
        artists_query  = f"""
            select distinct artist from (SELECT
                parts[3]::VARCHAR AS artist
            FROM ({self.unique_tracks_db}))
        """

        artists = duckdb.query(artists_query).to_df()['artist'].to_numpy()
        genres  = self.get_genres()
    
        artist_encoder = LabelEncoder().fit(artists)
        genre_encoder = LabelEncoder().fit(genres)

        del artists, genres

        artist_genre_query = f"""
            select mtd.track_id, mtd.genre, ut.artist from (select 
                parts[1]::VARCHAR AS track_id,
                parts[3]::VARCHAR AS artist
            from ({self.unique_tracks_db})
            ) as ut
            join {self.mds_tagtraum_db} as mtd on  ut.track_id = mtd.track_id
        """
        df = duckdb.query(artist_genre_query).to_df()

        artist_tensor = torch.tensor(artist_encoder.transform(df['artist'])).long()
        genre_tensor = torch.tensor(genre_encoder.transform(df['genre'])).long()

        dataloader = DataLoader(
            TensorDataset(artist_tensor, genre_tensor),
            batch_size=16,
            shuffle=True
        )

        encoder = SongEncoder(
            num_artists=len(artist_encoder.classes_),
            num_genres=len(genre_encoder.classes_)
        )

        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(6):
            encoder.train()
            total_loss = 0

            for artists, genres in dataloader:

                optimizer.zero_grad()

                _, artist_logits, genre_logits = encoder(artists, genres)

                loss_artist = criterion(artist_logits, artists)
                loss_genre  = criterion(genre_logits, genres)

                loss = loss_artist + loss_genre

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: {total_loss/len(dataloader):.4f}")


        encoder.eval()
        song_vectors = []
        with torch.no_grad():
            for artists, genres in dataloader:
                
                _, artist_logits, genre_logits = encoder(artists, genres)
                song_vectors.append(torch.concat([artist_logits, genre_logits]).numpy())

        
        np.save("songs_embedding.npy", np.array(song_vectors, dtype=np.float32))
        


    def content_based_recommendation(self, user_profile_df, baseline=True):


        song_vectors = np.load("songs_embedding.npy")

        faiss.normalize_L2(song_vectors)

        index = faiss.IndexFlatIP(song_vectors.shape[1])
        index.add(song_vectors)
        ## encoder user profile
        ## then find the top similar items!

        # top_idx = None

        # if baseline:
        #     items_train, items_test, indices_train, indices_test = self.items_matrix_split(user_profile_df)
        #     play_count_train =  user_profile_df['play_count'].iloc[indices_train].to_numpy()
        #     user_taste = np.sum(play_count_train[:, None] * items_train, axis=0) / np.sum(play_count_train)
        #     sim = cosine_similarity(user_taste.reshape(1, -1), items_test)
            
        #     flat = sim.ravel()

        #     top_idx = np.argpartition(flat, -10)[-10:]
        #     top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
        #     return user_profile_df.iloc[top_idx].sort_values(by='play_count', ascending=False)


        # test_df = self.user_profile_df(limit=100)
        # user_profile_df = user_profile_df.sample(frac=1, random_state=42)

        


        # n_artists = len(user_profile_df['artist'].unique())
        # n_genres  = len(user_profile_df['genre'].unique())

        # labels  = torch.tensor(
        #     np.where(user_profile_df['play_count'] >= 15, 1, 0).astype(np.float32)
        # ).unsqueeze(1)                                          # shape: (N, 1)

        # artists = torch.from_numpy(
        #     LabelEncoder().fit_transform(user_profile_df['artist'])
        # ).long()

        # genres  = torch.from_numpy(
        #     LabelEncoder().fit_transform(user_profile_df['genre'])
        # ).long()

        # dataset = TensorDataset(artists, genres, labels)

        # train_idx, test_idx = train_test_split(
        #     np.arange(len(dataset)),
        #     test_size=0.2,
        #     stratify=labels.squeeze().numpy(),                  # ← fix: squeeze + numpy
        #     random_state=42
        # )

        # train_dataset = Subset(dataset, train_idx)
        # test_dataset  = Subset(dataset, test_idx)

        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=16,
        #     shuffle=True                                        # ← fix: shuffle train
        # )

        # cbn       = ContentBasedNN(num_artists=n_artists, num_genres=n_genres)
        # optimizer = torch.optim.Adam(cbn.parameters(), lr=0.001)
        # criterion = nn.BCELoss()

        # for epoch in range(30):
        #     cbn.train()
        #     total_loss = 0

        #     for artists, genres, labels in train_loader:
        #         optimizer.zero_grad()
        #         predictions = cbn(artists, genres)
        #         loss        = criterion(predictions, labels)
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()

        #     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # cbn.eval()
        # correct = 0
        # total   = 0

        # with torch.no_grad():
        #     for artists, genres, labels in test_loader:

        #         outputs = cbn(artists, genres)
        #         preds   = torch.round(outputs).squeeze()
        #         labels  = labels.squeeze()
        #         correct += (preds == labels).sum().item()
        #         # print((preds == 1))
        #         # print(labels)
        #         # print(preds)
        #         # indices = test_idx[passed_batch:len(labels) + passed_batch]

        # #         np.ar(preds == 1).numpy()      # ← fix: round for binary
        #         # labels  = labels.squeeze()                      # ← fix: squeeze labels

        #         total   += labels.size(0)
        #         # passed_batch += len(labels)

        # accuracy = correct / total
        # print(f"Accuracy: {accuracy:.4f}")
        
       