from rec_sys import Recommender


def main():

    try:
        recommender = Recommender()
        user_id = '02a4255f067037ab82375a12d941a3df6ba93248'
    
        # recommender.top_250_tracks()
        # recommender.top_100_tracks_by_genre()
        # recommender.collections()
        recommender.user_based_recommendation(user_id)







    except Exception as e:
        print(f"error: {str(e)}")

if __name__ == "__main__":
    main()