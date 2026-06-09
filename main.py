import argparse
from rec_sys import Recommender
import sys


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

    recommender.top_250_tracks()
    recommender.top_100_tracks_by_genre()
    recommender.collections(process=process)
    recommender.user_based_recommendation(user_id)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        sys.exit(1)