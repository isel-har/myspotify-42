# from pandasql import sqldf
# import polars as pl
import gensim.downloader as api
import pandas as pd
import duckdb
# import gc

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
                                    columns={'line':'VARCHAR'})
        )"""

        self.unique_tracks_db = f"""SELECT string_split(line, '<SEP>') AS parts
                         FROM {self.unique_tracks}"""


        
# Download and load a small Word2Vec model
        self.w2v_model = api.load("glove-wiki-gigaword-50")


    def triplets_tracks_db(self, columns):
        return f"""
            SELECT {columns}
            FROM
            {self.train_triplets_db} as sdb
            JOIN
            (
                    SELECT
                         parts[1]::VARCHAR AS track_id,
                         parts[2]::VARCHAR AS song_id,
                         parts[3]::VARCHAR AS artist,
                         parts[4]::VARCHAR AS title
                     FROM ({self.unique_tracks_db}
            )  AS tdb
            ON sdb.song_id=tdb.song_id
        """


    def top_250_tracks(self):

        result = duckdb.sql(f"""{self.triplets_tracks_db("tdb.artist, tdb.title, sdb.play_count")}
            ORDER by play_count desc
            limit 250
        """)
        print("_____________________Top-250 tracks_____________________")
        print(result)
        del result


    def top_100_tracks_by_genre(self):

        genres = list(duckdb.query(f"select genre from {self.mds_tagtraum_db} group by genre").to_df().values.ravel())

        cte = f"""with tracks_table as ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')})"""
        
        query_ = f"""{cte} select tt.artist, tt.title, tt.play_count from tracks_table as tt
                join {self.mds_tagtraum_db} as mds on tt.track_id = mds.track_id"""
        

        for genre in genres:
            
            result = duckdb.query(f"""{query_}
                where mds.genre like '{genre}'
                order by tt.play_count desc
                limit 100
            """)

            print(f"_______________[Top 100 of {genre}]_______________")
            print(result)


    def collection(self, theme, keywords, threshold=0.05):
        
        collection = {
            'track_id':[],
        }

        theme_index = []
        for val in self.themes[theme]:
            try:
                index = keywords.index(val)
                theme_index.append(index)
            except ValueError:
                continue

        with open('data/mxm_dataset_train.txt', 'r') as f:
            for line in f:
                if line.startswith(('%', '#')):
                    continue

                parts = line.strip().split(',')
                track_id = parts[0]

                theme_score  = 0
                total_words = 0

                for part in parts[2:]:
                    word_index, count = part.split(':')
                    word_index = int(word_index)
                    count = int(count)

                    total_words += count

                    if word_index in theme_index:
                        theme_score += count

                theme_ratio = theme_score / total_words if total_words else 0

                if theme_ratio > threshold:
                    collection['track_id'].append(track_id)


        theme_db = duckdb.from_df(pd.DataFrame(data=collection))
        result = duckdb.query(f"""
                    select tt.artist, tt.title, tt.play_count from
                    ({self.triplets_tracks_db('tdb.artist, tdb.title, sdb.play_count, tdb.track_id')}) as tt
                    join theme_db tm
                    on tt.track_id=tm.track_id
                    order by tt.play_count desc
                    limit 50
                """)
        return result


    def collections(self):
        
        keywords = pd.read_csv('data/mxm_dataset_train.txt', comment='#', nrows=1) \
            .columns.to_list()

        print("Baseline")
        for theme in self.themes:
            print(f"_______________[Top 50 {theme}]_______________")
            collection = self.collection(theme, keywords)
            print(collection)
            del collection





    def word_vec(self):
        
        for theme in self.themes:
            top_n = 10
            if theme in self.w2v_model:
                similar_words = self.w2v_model.most_similar(theme, topn=top_n)
                for w, score in similar_words:
                    print(f"{w} -> {score}")
            else:
                print(f"{theme} not in vocabulary")



    def ten_similar(self):
        return

    def ten_similar_track(self):
        return



def main():

    try:
        recommender = Recommnender()

        recommender.word_vec()
        # recommender.top_250_tracks()
        # recommender.top_100_tracks_by_genre()
        # recommender.collections()

    except Exception as e:
        print(f"error: {str(e)}")

if __name__ == "__main__":
    main()