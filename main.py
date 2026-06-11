import argparse
from tools.rec_sys import Recommender
from tools.tools import train_test_split_matrix
import sys
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description='rec sys')

    parser.add_argument(
        "--process",
        "-p",
        action="store_true"
    )

    parser.add_argument(
        "--user_id",
        "-uid",
        type=str,
        default='02a4255f067037ab82375a12d941a3df6ba93248'
    )

    args = parser.parse_args()

    return args.process, args.user_id


def main():

    process, user_id = parse_args()
    recommender = Recommender()

    # themes = ['love', 'war','happiness']

    # df = recommender.top_250_tracks()
    # print("Top-250 tracks")
    # print(df)

    # genres = recommender.get_genres()
    # for genre in genres:

    #     df = recommender.top_100_tracks_by_genre(genre)
    #     print(f"top 100 of genre '{genre}'")
    #     print(df)


    # print("baseline approach")
    # for theme in themes:
    #     df = recommender.collections(theme=theme)
    #     print(df)

    # print("Word2vec approach")
    # for theme in themes:
    #     df = recommender.collections(theme=theme, approach='word2vec')
    #     print(df)

    # print("classification approach")
    # for theme in themes:
    #     df = recommender.collections(
    #         theme=theme,
    #         process=process,
    #         approach='classification'
    #     )
    #     print(df)


    # user_item_matrix = recommender.user_item_matrix()
    # train_matrix, test_matrix = train_test_split_matrix(user_item_matrix)

    # top_10, p_at_10 = recommender.user_based_recommendation(user_id, train_matrix, test_matrix)
    # print("cosine similarity approach")
    # print("Average p@10:", p_at_10)
    # print("top 10 recommendation:")
    # print(top_10)

    # top_10, p_at_10 = recommender.user_based_recommendation(user_id, train_matrix, test_matrix, False)
    # print("Matrix factorization approach")
    # print("Average p@10:", p_at_10)
    # print("top 10 recommendation:")
    # print(top_10)


    print("Bonus part")
    print("Content-based cosine similarity approach")

    user_profile_df = recommender.user_profile_df(user_id)
    top_idx = recommender.content_based_recommendation(user_profile_df)
    top_10 = user_profile_df.iloc[top_idx].sort_values(by='play_count', ascending=False)
    print(top_10)


    # top_10 = recommender.content_based_recommendation(user_taste, items_test, False)
    # print(top_10)# add p@10


    # train_loader, test_loader = recommender. dataloader_train_test_split()
    # top_10 = recommender.ncf_recommendation(train_loader, test_loader)
    # print(top_10)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"exception : {str(e)}")
        sys.exit(1)