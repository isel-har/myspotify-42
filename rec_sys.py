import os
os.environ['GENSIM_DATA_DIR'] = '/Users/isel-har/goinfre/gensim'

import numpy as np
import pandas as pd
import duckdb
from sklearn.naive_bayes import MultinomialNB
import gensim.downloader as api
from nltk.corpus import stopwords
# from scipy.sparse import csr_matrix
from tools import train_test_split_matrix, compute_item_similarity, recommend, precision_at_k
from tools import train_svd, recommend_svd, compute_scores, precision_at_k_svd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib



# pd.set_option('display.max_rows', None)
import nltk
nltk.download('stopwords')


class Recommender:
    def __init__(self, load_word2vec=False):
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


        self.stop_words = set(stopwords.words("english"))
        
        self.w2v_model  = api.load('glove-twitter-25') if load_word2vec else None


    # ROW_NUMBER() OVER (ORDER BY tdb.track_id) as index,
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

        print("_____________________Top-250 tracks_____________________")
        print(result)
        del result


    def top_100_tracks_by_genre(self):

        genres = duckdb.query(f"""
            select distinct genre
            from {self.mds_tagtraum_db}
        """).to_df()['genre'].tolist()

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

        for genre in genres:

            result = duckdb.query(f"""
                {base_query}
                WHERE mds.genre like '{genre}'
                ORDER BY tt.play_count DESC
                LIMIT 100
            """)

            print(f"_______________[Top 100 of {genre}]_______________")
            print(result)

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
            similar_tokens = self.word_vec(theme=theme, top_n=top_n)

            for token, score in similar_tokens:
                # if score < 0.8:
                #     continue
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




    def preprocessing(self, test_size=0.2):
        
        df = None
        for theme in self.themes:
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
            test_size=test_size,
            stratify=df['theme'],
            random_state=42
        )

        le   = LabelEncoder()
        mx_dict = self.mxm_dict()

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
        
        X_train, X_test, y_train, y_test = joblib.load('data/X_train.pkl'), joblib.load('data/X_test.pkl'), \
            joblib.load('data/y_train.pkl'), joblib.load('data/y_test.pkl')


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

        y_pred = clf.predict(X_test)

        print(f"accuracy {accuracy_score(y_true=y_test, y_pred=y_pred)}")
        
        




    def collection_classification(self, theme, model, theme_index, threshold=0.7):

        collection = {'track_id': []}
        with open('data/mxm_dataset_train.txt', 'r') as f:

                for line in f:

                    if line.startswith(('%', '#')):
                        continue

                    parts = line.strip().split(',')
                    track_id = parts[0]

                    vec = np.zeros(self.vocabulary_size)
                    for part in parts[2:]:

                        word_index, count = part.split(':')
                        word_index        = int(word_index)
                        count             = int(count)
                        vec[word_index - 1] = count   
                    
                    pred_ratios = model.predict_proba([vec])
                    pred_theme = model.predict([vec])


                    if pred_ratios[0][theme_index] >= threshold and  pred_theme[0] == self.class_labels[theme]:
                        collection['track_id'].append(track_id)


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
        return result


    def collections(self):
        
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

        # print("baseline approache")
        # for theme in self.themes:
        #     print(f"theme : {theme}")
        #     collection = self.collection(theme, threshold=0.065, min_theme_words=5)
        #     print(collection)
        #     del collection
        
        # print("word2vec approache")
        # for theme in self.themes:
        #     print(f"theme : {theme}")
        #     collection = self.collection(theme, threshold=0.07, word2vec=True, top_n=10, min_theme_words=5)
        #     print(collection)
        #     del collection

        # self.preprocessing()
        # print("classification approache")## slow and not accuracte!
        # model = self.classification()
        # for i, theme in enumerate(self.themes):
        #     print(f"theme : {theme}")
        #     collection = self.collection_classification("love", model, i)
        #     print(collection)
        #     del collection


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



    def user_based_recommendation(self, user_id=''):

        user_item_matrix          = self.user_item_matrix()
        train_matrix, test_matrix = train_test_split_matrix(user_item_matrix)

        top_10_rec, p_at_10  = self.cosine_similarity_approach(user_id, train_matrix, test_matrix)()

        # print("cosine similarity approach")
        # print("Average p@10:", p_at_10)
        # print("top 10 recommendation:")
        # print(top_10_rec)


        # top_10_rec, p_at_10 = self.matrix_factorization_approach(user_id, train_matrix, test_matrix)

        # print("SVD approach")
        # print("Average p@10:", p_at_10)
        # print("top 10 recommendation:")
        # print(top_10_rec)
