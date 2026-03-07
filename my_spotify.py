import pandas as pd
import numpy as np
# import sys
import gc

class Recommnender:
    def __init__(self):
        ...

    def top_250_tracks(self):

        count_df = pd.read_csv('data/train_triplets.txt',
            usecols=[1, 2],
            delimiter='\t',
            names=['song_id', 'count']
        ) \
        .sort_values(by='count', ascending=False) \
        .head(250)

        metadata_df = pd.read_csv(
            'data/p02_unique_tracks.txt',
            usecols=[0, 1, 2, 3],
            delimiter='<SEP>',
            names=['track_id', 'song_id', 'artist', 'title'],
            engine='python'
        )
        
        top_250_df = pd.merge(count_df, metadata_df, on='song_id')

        print(top_250_df[['song_id', 'artist', 'title']].head(250))

        del count_df
        del metadata_df

        return top_250_df

    def top_100_tracks_by_genre(self):

        top_250_df = self.top_250_tracks()

        genre_df = pd.read_csv('data/p02_msd_tagtraum_cd2.cls',
                sep='\t',
                engine='python-fwf',
                comment='#',
                usecols=[1, 2],
                names=['track_id', 'majority-genre']
        )

        df = pd.merge(top_250_df, genre_df, on='track_id') \
            .groupby(by='majority-genre') \
            .head(100)
    
        print(df[['artist', 'title', 'count']])

        del df
        del genre_df
        del top_250_df




    def collections(self):
        return
    
    def ten_similar(self):
        return

    def ten_similar_track(self):
        return



def main():

    try:
        recommender = Recommnender()

        # recommender.top_250_tracks()
        recommender.top_100_tracks_by_genre()


    except Exception as e:
        print(f"error: {str(e)}")

if __name__ == "__main__":
    main()