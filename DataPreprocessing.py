import numpy as np
import DescriptiveStats as ds
import time
import pickle
import KeywordMatrix as km
import UserMatrix as um
import pandas as pd
import PredictiveModels as mm


def gen_user_ratings():
    ratings = np.genfromtxt('ml-100k/u.data', delimiter='\t', dtype=None)

    test = np.zeros((943,1682))
    for x in range(0,100000):
        user_id = ratings[x][0]
        item_id = ratings[x][1]
        rating = ratings[x][2]
        test[user_id-1][item_id-1] = rating

    np.savetxt("postprocessed-data/user_ratings",test, delimiter=",", fmt='%1d')


def gen_grouplens_movie_list():
    movies = np.genfromtxt('ml-100k/u.item', delimiter='|', usecols=(1), dtype=None)
    output = open("ml-100k/movie_list", "w")
    for item in movies:
        if not item[0] == "\"":
            print>>output, item


def gen_movies_with_keywords_before_1999():
    import re
    movie_keywords_file = open("imdb/keywords.list")
    movie_keywords_lines = movie_keywords_file.readlines()[75783:]


    output = open("imdb/movie_keywords.list", "w")

    for item in movie_keywords_lines:
        #year = int(item.split("(")[1][0:4])
        year = re.search('(\d{4})|(\?\?\?\?)', item).group(0)
        if not item[0] == "\"" and not year == "????" and int(year) < 1999 and not re.search('(\(TV\))|(\(V\))|(\(VG\))',item):
            print>>output, item.strip()


def match_grouplens_with_imdb():
    #import re
    grouplens_movie_list = open("ml-100k/movie_list")
    grouplens_lines = grouplens_movie_list.read().splitlines()

    imdb_movie_keywords = np.genfromtxt('imdb/movie_keywords_no_hashtags', delimiter = "\t", dtype = str)

    grouplens_lines_set = set(grouplens_lines)

    '''
    alt_set = set()

    for title in grouplens_lines:
        if title[-8] == ')':
            alt_title = re.search('\((\S.+)\)', title[:-7]).group(0)[1:-1]
            alt_line = alt_title + title[-7:]
            alt_set.add(alt_line)
            inner_line = title[:-(len(alt_line)+3)] + title[-7:]
            alt_set.add(inner_line)
    '''

    output = open("imdb/movies_in_list", "w")
    for line in imdb_movie_keywords:
        title = line[0]
        keyword = line[1]
        alt_line = title
        if title[:4] == 'The ':
            alt_line = title[4:-7] + ', The' + title[-7:]
        elif title[:2] == 'A ':
            alt_line = title[2:-7] + ', A' + title[-7:]
        if alt_line in grouplens_lines_set:
                movie = str(grouplens_lines.index(alt_line)+1) + '\t' + alt_line + '\t' + title + '\t' + keyword
                print>>output, movie


def sort_movie_list():
    movies = np.genfromtxt('imdb/movies_in_list', delimiter = '\t', dtype = None)
    movies = np.sort(movies, axis=0)
    np.savetxt("sorted_movies_in_list", movies, delimiter="\t", fmt="%s")


def gen_all_keyword_list():
    movies = np.genfromtxt('imdb/movies_in_list', delimiter = '\t', dtype = str, usecols=3)
    keywords_set = set()
    for item in movies:
        keywords_set.add(item)

    output = open("imdb/keywords_set", "w")

    for item in keywords_set:
        print>>output, item

'''
user_id = 229
infile = open("postprocessed-data/all_user_matrix_actress_and_actor", "r")
all_user_matrix = pickle.load(infile)
results = model_all_users(all_user_matrix, [1,10,15])
print results[0]
print np.average(results[0])

ridge_mae_table = model_all_users(all_user_matrix, [15])
ridge_mae_table.to_csv(path_or_buf = "tables/ridge_mae_table.csv", index = 0)

'''

'''
infile = open("postprocessed-data/all_user_matrix_num_ratings", "r")
all_user_matrix = pickle.load(infile)

user_id = 38
ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)

user_matrix = all_user_matrix[user_id-1]
user_ratings = ratings[user_id-1]

results = model_user(user_matrix)
movie_id = 432


gen_sig_features(user_id, results, user_matrix, user_ratings)
'''

'''
df = user_results = gen_user_results(user_id, results)
df.to_csv(path_or_buf = "tables/user_" + str(user_id) + "_results.csv", index = 0)


df = movie_weights = gen_movie_weights(movie_id,results,user_matrix)
df.to_csv(path_or_buf = "tables/user_" + str(user_id) + "_movie_" + str(movie_id) + "_weights.csv", header = ["Feature", "Weights"], index = 0)
'''

'''
import numpy as np
user_id = 944

print "Loading user matrix"
user_matrix = um.get_user_matrix(user_id)

print "Loading user ratings and movies"
ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=int)
movies = np.genfromtxt('ml-100k/movie_list', delimiter='|', dtype=str)
user_ratings = ratings[user_id-1]

from Classifiers import TransparentRidge

clf =TransparentRidge(alpha=0.001)
user_cols = user_matrix.shape[1]
data = user_matrix[:, 1:(user_cols-1)]
target = user_matrix[:, (user_cols-1)]
clf.fit(data,target)
weights = clf.coef_
neg_evi, pos_evi = clf.predict_evidences(data)
bias = clf.get_bias()
print bias
y_pred = clf.predict(data)
indices = np.argsort(y_pred)

#The Highest Rating
j = indices[-1]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,weights,user_matrix)
print "Movie Title: ", movies[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)

#The Lowest Rating
j = indices[0]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,weights,user_matrix)
print "Movie Title: ", movies[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)

# The case that has the most negative evidence, regardless of positive evidence
j = np.argsort(neg_evi)[0]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,weights,user_matrix)
print "Movie Title: ", movies[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)

# The case that has the most positive evidence, regardless of negative evidence
j = np.argsort(pos_evi)[-1]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,weights,user_matrix)
print "Movie Title: ", movies[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)

# Most conflicted
conflict = np.min([abs(neg_evi), pos_evi], axis=0)
indices = np.argsort(conflict)
j=indices[-1]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,weights,user_matrix)
print "Movie Title: ", movies[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)

# Least amount of info
information = np.max([abs(neg_evi), pos_evi], axis=0)
indices = np.argsort(information)
j=indices[0]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,weights,user_matrix)
print "Movie Title: ", movies[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)
'''
