import numpy as np
from sklearn import preprocessing as pp
from sklearn.metrics import mean_absolute_error
import pickle


def get_user_avg_rating_mae(ratings, user_id):
    user_ratings = ratings[user_id-1]
    avg_ratings = get_avg_ratings_for_user(ratings, user_id)
    user_ratings = user_ratings [np.nonzero(user_ratings )]
    mae = mean_absolute_error(user_ratings, avg_ratings)
    return mae


def get_avg_ratings_for_user(ratings, user_id):
    user_ratings = ratings[user_id-1]
    avg_ratings = []
    for movie in range(0, 1681):
        if user_ratings[movie] != 0:
            avg = get_avg_rating_for_movie(ratings, movie)[0]
            avg_ratings.append(avg)
    return avg_ratings


def get_avg_rating_for_movie(ratings, movie):
    movie_ratings = np.array(ratings[:,movie])
    nonzero_movie_ratings = movie_ratings[np.nonzero(movie_ratings)]
    num_movie_ratings = len(nonzero_movie_ratings)
    avg_rating = float(np.sum(nonzero_movie_ratings))/float(len(nonzero_movie_ratings))
    return (avg_rating, num_movie_ratings)


def get_avg_ratings(ratings):
    avg_ratings = []
    num_ratings = []
    for x in range(0, 1681):
        result = get_avg_rating_for_movie(ratings, x)
        avg_ratings.append(result[0])
        num_ratings.append(result[1])
    return np.column_stack((avg_ratings,num_ratings))


# Get the genre and ratings matrix for user
def get_user_matrix_with_movie_index(user_id, movies, ratings):
    user = np.transpose(ratings[user_id - 1])

    user_set = np.zeros((0, 23))

    for x in range(0, 1681):
        rating = user[x]
        if rating > 0:
            results = get_avg_rating_for_movie(ratings, x)
            avg_rating = results[0]
            num_rating = results[1]
            temp = np.append(x + 1, movies[x])
            temp = np.append(temp, avg_rating)
            temp = np.append(temp, num_rating)
            new_row = np.append(temp, rating)
            user_set = np.vstack((user_set, new_row))

    return user_set


# Combine the genre, keywords, and ratings into one matrix per user
def get_concat_ratings_keywords(user_set, keyword_matrix):
    user_cols = user_set.shape[1]
    keyword_cols = keyword_matrix.shape[1]
    user_set_full= np.empty((0,user_cols + keyword_cols))
    for row in user_set:
        temp = np.append(row[0:(user_cols -1)], keyword_matrix[row[0] - 1])
        new_row = np.append(temp, row[user_cols-1])
        user_set_full = np.vstack((user_set_full, new_row))
    return user_set_full


def get_concat_genre_keyword_actor(user_set):
    actor_matrix = np.genfromtxt('postprocessed-data/actor_and_actress_matrix', delimiter=',')
    user_cols = user_set.shape[1]
    actor_cols = actor_matrix.shape[1]
    user_set_full= np.empty((0,user_cols + actor_cols))
    ratings = user_set[:,-1]
    indicies = np.array(np.transpose(user_set[:,0]), dtype=int) - 1
    actor_matrix = actor_matrix[indicies]
    user_set_full = np.column_stack((user_set[:,0:-1], actor_matrix, ratings))
    return user_set_full


def gen_all_user_matrix():
    import pickle
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
    movies = np.genfromtxt('ml-100k/u.item', delimiter='|', usecols=range(5, 24))
    keyword_matrix = np.genfromtxt('postprocessed-data/keywords_matrix', delimiter=',', dtype=int)
    num_users = ratings.shape[0]
    all_user_matrix = [None] * num_users

    for user_id in range(1,num_users+1):
        user_matrix = get_user_matrix_with_movie_index(user_id, movies, ratings)
        user_set_genre_keyword = get_concat_ratings_keywords(user_matrix, keyword_matrix)
        user_matrix_full = get_concat_genre_keyword_actor(user_set_genre_keyword)
        pp.scale(user_matrix_full[:,20], axis=0, with_mean=True, with_std=True, copy=False)
        pp.scale(user_matrix_full[:,21], axis=0, with_mean=True, with_std=True, copy=False)
        all_user_matrix[user_id-1] = user_matrix_full

    outfile = open("postprocessed-data/all_user_matrix_num_ratings", "w")
    pickle.dump(all_user_matrix, outfile)


def get_all_user_matrix():
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
    movies = np.genfromtxt('ml-100k/u.item', delimiter='|', usecols=range(5, 24))
    keyword_matrix = np.genfromtxt('postprocessed-data/keywords_matrix', delimiter=',', dtype=int)
    num_users = ratings.shape[0]
    all_user_matrix = [None] * num_users

    for user_id in range(1,num_users+1):
        all_user_matrix[user_id-1] = get_user_matrix(user_id,ratings,movies,keyword_matrix)

    return all_user_matrix


def gen_all_movie_matrix():
    movies = np.genfromtxt('ml-100k/u.item', delimiter='|', usecols=range(5, 24))[0:-1]
    keyword_matrix = np.genfromtxt('postprocessed-data/keywords_matrix', delimiter=',', dtype=int)
    actor_matrix = np.genfromtxt('postprocessed-data/actor_and_actress_matrix', delimiter=',')
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)

    movie_ids = np.array(range(1,1682))

    avg_num_ratings = get_avg_ratings(ratings)

    movie_matrix = np.column_stack((movie_ids,movies,avg_num_ratings, keyword_matrix,actor_matrix))

    #np.savetxt("postprocessed-data/movie_matrix", movie_matrix, delimiter=",", fmt='%.8f')
    outfile = open("postprocessed-data/movie_matrix", "w")
    pickle.dump(movie_matrix, outfile)


def get_user_matrix1(user_id):
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
    movies = np.genfromtxt('ml-100k/u.item', delimiter='|', usecols=range(5, 24))
    keyword_matrix = np.genfromtxt('postprocessed-data/keywords_matrix', delimiter=',', dtype=int)

    user_set = get_user_matrix_with_movie_index(user_id, movies, ratings)
    user_set_genre_keyword = get_concat_ratings_keywords(user_set, keyword_matrix)
    user_matrix = get_concat_genre_keyword_actor(user_set_genre_keyword)

    pp.scale(user_matrix[:,20], axis=0, with_mean=True, with_std=True, copy=False)
    pp.scale(user_matrix[:,21], axis=0, with_mean=True, with_std=True, copy=False)
    return user_matrix


infile = open("postprocessed-data/movie_matrix", "r")
movie_matrix = pickle.load(infile)
infile.close()


def get_user_matrix(user_id, ratings):
    user_ratings = ratings[user_id-1]
    user_movies = np.nonzero(user_ratings)

    user_matrix = np.column_stack((movie_matrix[user_movies], user_ratings[user_movies]))
    pp.scale(user_matrix[:,20], axis=0, with_mean=True, with_std=True, copy=False)
    pp.scale(user_matrix[:,21], axis=0, with_mean=True, with_std=True, copy=False)
    return user_matrix