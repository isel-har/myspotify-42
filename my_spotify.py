# from pandasql import sqldf
# import polars as pl
import gensim.downloader as api
import pandas as pd
import numpy as np
import duckdb

# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB


class Recommnender:
    def __init__(self):
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


        
        self.w2v_model = api.load("glove-wiki-gigaword-50")
        # self.stop_words = set(stopwords.words("english"))


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


    def collection(self, theme, threshold, word2vec=False):

        collection = {
            'track_id':[],
        }

        theme_index = set()

        for val in self.themes[theme]:
            try:
                index = self.keywords.index(val)
                theme_index.add(index)
                        
            except ValueError:
                continue
    
        if word2vec:
            similar_tokens = self.word_vec(theme=theme, top_n=10)
            for token, n in similar_tokens:
                try:
                    index = self.keywords.index(token)
                    theme_index.add(index)
                except ValueError:
                    continue

        with open('data/mxm_dataset_train.txt', 'r') as f:
            for line in f:
                if line.startswith(('%', '#')):
                    continue

                parts = line.strip().split(',')
                track_id = parts[0]

                theme_score  = 0
                total_words  = 0

                for part in parts[2:]:
                    word_index, count = part.split(':')
                    word_index = int(word_index)
                    count      = int(count)

                    total_words += count

                    if word_index in theme_index:
                        theme_score += count

                theme_ratio = theme_score / total_words if total_words else 0

                if theme_ratio > threshold:
                    collection['track_id'].append(track_id)

        data_frame = pd.DataFrame(data=collection)
        # theme_db = duckdb.from_df(data_frame)
        # result = duckdb.query(f"""
        #     select 
        #         tt.artist,
        #         tt.title,
        #         sum(tt.play_count) as play_count
        #     from
        #         ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')}) as tt
        #     join theme_db tm
        #         on tt.track_id = tm.track_id
        #     group by 
        #         tt.track_id, tt.artist, tt.title
        #     order by play_count desc
        #     limit 50
        # """)

        if word2vec: 
            try:
                print("Data frame size:",len(data_frame))
                n_sample = 1000 if len(data_frame) >= 1000 else len(data_frame)
                print("max sample :",n_sample)
                sample = data_frame.sample(n=n_sample).copy()
                sample['TARGET'] = theme
                sample.to_csv(f"data/train_themes.csv", mode='a', index=False, header=False)

                del sample
            except Exception as e:
                print('error:', str(e))
        return None


    def fill_vec(self, track_pairs, voc_size):
        
        vec = np.zeros(voc_size)

        pairs = track_pairs.split(',')
        for pair in pairs:
            
            word_index, count = pair.split(':')
            word_index = int(word_index)
            count      = int(count)

            vec[word_index] = count
        return vec



    def classification(self):
        
        # x_batch = []
        # y_batch = []

        # voc_size = len(self.keywords)
        # classes  = {
        #     'love': np.float32(0.0),
        #     'war': np.float32(1.0),
        #     'happiness': np.float32(2.0),
        #     'loneliness': np.float32(3.0),
        #     'money':np.float32(4.0)
        # }
        
        df = pd.read_csv('data/train_themes.csv', header=None, names=['track_id', 'theme'])
        print("________________________________________________________________")
        print((len(df[df['theme'] == 'love'])))
        print((len(df[df['theme'] == 'war'])))
        print((len(df[df['theme'] == 'money'])))
        print((len(df[df['theme'] == 'loneliness'])))
        print((len(df[df['theme'] == 'happiness'])))
        # shuffled_df = df.sample(frac=1)
        # print(shuffled_df)

        # with open('data/train_themes.csv', 'r') as f:

        #     for line in f:
                
        #         parts = line.split(',')
        #         track_id = parts[0]
        #         category = parts[1]


        #         with open('data/mxm_dataset_train.txt', 'r') as ml:
        #             for mline in ml:
                        
        #                 if mline.startswith(('#', '%')):
        #                     continue

        #                 track__id = mline[: mline.find(',')]
        #                 if track_id == track__id:
        #                     start = mline.find(',',  mline.find(',', len(track__id)) + 1)
        #                     keywords_vec = "dwa"# self.fill_vec(mline[start:], voc_size)
        #                     x_batch.append(keywords_vec)
        #                     y_batch.append(classes[category])

        #                     break
        #         break
        
        # print(x_batch)
        # print(y_batch)


    def collections(self, threshold=0.05):
        
        self.keywords = pd.read_csv('data/mxm_dataset_train.txt', comment='#', nrows=1) \
            .columns.to_list()
        # print(len(keywords))
        # print("Baseline")
        # for theme in self.themes:
        #     print(f"_______________[Top 50 {theme}]_______________")
        #     collection = self.collection(theme, threshold=threshold)
        #     print(collection)
        #     del collection
        
        # print("Word2vec")
        
        for theme in self.themes:
            print(f"_______________[Top 50 {theme}]_______________")
            collection = self.collection(theme, threshold=threshold, word2vec=True)
            # print(collection)
            # del collection

        # print("Classification")
        self.classification()


    def ten_similar(self):## collaborative filtering
        return

    def ten_similar_track(self):
        return


def main():

    try:
        recommender = Recommnender()

        # recommender.top_250_tracks()
        # recommender.top_100_tracks_by_genre()
        recommender.collections(threshold=0.06)



    except Exception as e:
        print(f"error: {str(e)}")

if __name__ == "__main__":
    main()