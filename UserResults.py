import PredictiveModels as mm
import numpy as np
import pandas as pd
import DescriptiveStats as ds
import UserMatrix as um


def model_all_users(all_user_matrix, alpha):
    mae_list = []
    for user_matrix in all_user_matrix:
        #results = mm.get_ridge_cv_results(user_matrix, alpha)
        results = mm.get_lasso_cv_results(user_matrix, alpha)
        mae = results[0]
        mae_list.append(mae)
    mae_table = pd.DataFrame(np.column_stack((range(1,944), mae_list)), columns = ["User ID", "MAE"])
    mae_table = mae_table.sort("MAE", ascending = 1)
    return mae_table


def test_alphas(alphas, all_user_matrix):
    for alpha in alphas:
        mae_list = model_all_users(all_user_matrix, [alpha])
        print sum(mae_list) / float(len(mae_list))

    mae_list = model_all_users(all_user_matrix, [15])
    mae_list = model_all_users(all_user_matrix, [.1])
    print sum(mae_list) / float(len(mae_list))
    ds.plot_mae_list(mae_list)


def model_user(user_matrix, movie_id):
    results = mm.get_lasso_cv_results(user_matrix, [.1])
    #results = mm.get_ridge_cv_results(user_matrix, [15])
    return results


def get_feature_names():
    genres = np.genfromtxt('ml-100k/u.genre', delimiter='|', dtype=str, usecols=0)
    keywords = np.genfromtxt('postprocessed-data/keyword_list', dtype=str)
    actors = open("postprocessed-data/pop_actor_set")
    actors = actors.read().splitlines()
    col_names = np.append(genres, 'Average Rating')
    col_names = np.append(col_names, 'Number of Ratings')
    col_names = np.append(col_names, keywords)
    col_names = np.append(col_names, actors)

    return col_names


def get_weights_and_features(results):
    features = get_feature_names()
    weights = results[1];

    positive_weight_indices = weights > 0
    negative_weight_indices = weights < 0

    positive_features = np.column_stack((features[positive_weight_indices], weights[positive_weight_indices].astype(float)))
    negative_features = np.column_stack((features[negative_weight_indices], weights[negative_weight_indices].astype(float)))

    positive_df = pd.DataFrame(positive_features, columns = ["Features", "Weights"])
    positive_df = positive_df.convert_objects(convert_numeric=True)
    positive_df = positive_df.sort("Weights", ascending=0)

    negative_df = pd.DataFrame(negative_features, columns = ["Features", "Weights"])
    negative_df = negative_df.convert_objects(convert_numeric=True)

    negative_df = negative_df.sort("Weights", ascending=1)

    return (positive_df,negative_df)


def get_model_test_ratings(user_id, results):
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
    user_ratings = ratings[user_id-1]

    nonzero_indices = np.nonzero(user_ratings)
    user_ratings = user_ratings[nonzero_indices]

    avg_ratings = um.get_avg_ratings(ratings)
    avg_ratings = avg_ratings[nonzero_indices]

    user_data = np.column_stack((avg_ratings, user_ratings, results[2]))
    return user_data


def gen_user_results_old(user_id, results):
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
    user_ratings = ratings[user_id-1]
    y_pred = results[2]
    nonzero_indices = np.nonzero(user_ratings)
    avg_ratings = um.get_avg_ratings(ratings)
    avg_ratings = avg_ratings[nonzero_indices]
    user_ratings = user_ratings[nonzero_indices]
    nonzero = nonzero_indices[0]

    error_from_average = np.abs(avg_ratings-user_ratings)
    error_from_pred = np.abs(user_ratings-y_pred)

    col_names = ["Movie ID" ,"Average Rating","Error from Avg","User Rating","Model Prediction","Model Error"]

    col_avg = ["Average:", np.average(avg_ratings),np.average(error_from_average),np.average(user_ratings),np.average(y_pred),np.average(error_from_pred)]
    user_result = np.column_stack((nonzero, avg_ratings,error_from_average, user_ratings, y_pred, error_from_pred))
    user_result = user_result[np.argsort(user_result[:, 5])]
    user_result = np.vstack((user_result,col_avg))

    df = pd.DataFrame(user_result, columns = col_names)
    #df = df.convert_objects(convert_numeric=True)

    return df


def gen_user_results(user_id, results):
    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
    user_ratings = ratings[user_id-1]

    y_pred = results[2]

    y_test = results[3]
    test_ids = results[4].astype(int) -1

    nonzero_indices = np.nonzero(user_ratings)

    avg_ratings = um.get_avg_ratings(ratings)[:,0]
    avg_ratings = avg_ratings[test_ids]

    user_ratings = user_ratings[test_ids]

    error_from_average = np.abs(avg_ratings-user_ratings)
    error_from_pred = np.abs(user_ratings-y_pred)

    col_names = ["Movie ID" ,"Average Rating","Error from Avg","User Rating","Model Prediction","Model Error"]

    col_avg = ["Average:", np.average(avg_ratings),np.average(error_from_average),np.average(user_ratings),np.average(y_pred),np.average(error_from_pred)]
    user_result = np.column_stack((test_ids+1, avg_ratings,error_from_average, user_ratings, y_pred, error_from_pred))
    user_result = user_result[np.argsort(user_result[:, 5])]
    user_result = np.vstack((user_result,col_avg))

    df = pd.DataFrame(user_result, columns = col_names)
    #df = df.convert_objects(convert_numeric=True)

    return df


def gen_movie_weights(movie_id, dm, user_matrix):
    movie_index = np.where(user_matrix[:,0] == movie_id)
    movie_dm = dm[movie_index][0]

    feature_names = get_feature_names()

    positive_weights = movie_dm[movie_dm > 0]
    negative_weights = movie_dm[movie_dm < 0]

    positive_names = feature_names[movie_dm > 0]
    negative_names = feature_names[movie_dm < 0]

    positive_weights = np.column_stack((positive_names,positive_weights))
    positive_weights = positive_weights[np.argsort(positive_weights[:,1])][::-1]

    negative_weights = np.column_stack((negative_names,negative_weights))
    negative_weights = negative_weights[np.argsort(negative_weights[:,1])][::-1]

    positive_df = pd.DataFrame(positive_weights, columns = ["Feature", "Weights"])
    positive_df = positive_df.convert_objects(convert_numeric=True)

    negative_df = pd.DataFrame(negative_weights, columns = ["Feature", "Weights"])
    negative_df = negative_df.convert_objects(convert_numeric=True)

    pd.options.display.float_format = '{:,.4f}'.format

    return [positive_df, negative_df]


def gen_sig_features(user_id, y_pred, weights, user_matrix, user_ratings):
    nonzero = np.nonzero(user_ratings)

    user_ratings = user_ratings[nonzero]

    movie_id_list = nonzero[0] + 1

    error_from_pred = np.abs(user_ratings-y_pred)

    x = np.column_stack((movie_id_list, error_from_pred))
    x = x[np.argsort(x[:,1])]

    positive_feature_set = dict()
    negative_feature_set = dict()
    for row in x:
        error = row[1]
        if error < .5:
            movie_id = row[0]
            res = gen_movie_weights(movie_id, weights, user_matrix)
            positive = res[0]
            negative = res[1]
            for feature in positive.head():
                    positive_feature_set[feature] = positive_feature_set.get(feature,0) + 1
            for feature in negative.head():
                    negative_feature_set[feature] = negative_feature_set.get(feature,0) + 1

    print positive_feature_set
    print negative_feature_set
    positive_df = pd.DataFrame(np.column_stack((positive_feature_set.keys(), positive_feature_set.values())), columns = ["Feature", "Times Largest Weight"])
    positive_df = positive_df.sort("Times Largest Weight", ascending = 0)
    print positive_df
    #positive_df.to_csv(path_or_buf = "tables/user_" + str(user_id) + "_pos_sig_features.csv", index = 0)
    print "\n"
    negative_df = pd.DataFrame(np.column_stack((negative_feature_set.keys(), negative_feature_set.values())), columns = ["Feature", "Times Lowest Weight"])
    negative_df = negative_df.sort("Times Lowest Weight", ascending = 0)
    print negative_df
    #negative_df.to_csv(path_or_buf = "tables/user_" + str(user_id) + "_neg_sig_features.csv", index = 0)


'''

import numpy as np
import UserMatrix as um
import pickle
from Classifiers import TransparentRidge
import DescriptiveStats as ds
import matplotlib
import matplotlib.pyplot as plt

# Loading user ratings and movie list
ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=int)
movies = np.genfromtxt('postprocessed-data/movie_list', delimiter='|', dtype=str)

# Loading user matrix
user_id = 945
user_matrix = um.get_user_matrix(user_id)
user_ratings = ratings[user_id-1]

# Creating model
clf =TransparentRidge(alpha=.05)
user_cols = user_matrix.shape[1]
data = user_matrix[:, 1:(user_cols-1)]
target = user_matrix[:, (user_cols-1)]
clf.fit(data,target)
weights = clf.coef_
neg_evi, pos_evi = clf.predict_evidences(data)
bias = clf.get_bias()
y_pred = clf.predict(data)
indices = np.argsort(y_pred)

# Predict all Films
infile = open("postprocessed-data/movie_matrix", "r")
movie_matrix = pickle.load(infile)
all_pred = clf.predict(movie_matrix[:, 1:])
movie_df = pd.DataFrame(np.column_stack((movies, all_pred)), columns = ["Movie", "Rating"])
movie_df = movie_df.sort("Rating", ascending = 0)
print movies[63]
print all_pred[63]
print movie_df.head(10)
exit(0)

from scipy.sparse import diags
coef_diags = diags(clf.coef_, 0)
dm = data * coef_diags

# The Highest Rating
j = indices[-1]
movie_id = user_matrix[j][0]
res = um.get_avg_rating_for_movie(ratings, movie_id-1)
avg_rating = res[0]
num_rating = res[1]
movie_features = gen_movie_weights(movie_id,dm,user_matrix)

print "Movie Title: ", movies[movie_id-1]
print "User Rating: ", user_ratings[movie_id-1]
print "Average Rating: ", avg_rating
print "Number of Ratings: ", num_rating
print "Prediction: ",  clf.predict(data[j])[0]
print "Bias and evidences:", bias, neg_evi[j], pos_evi[j]
print "Positive Features"
print movie_features[0].head(10)
print "Negative Features"
print movie_features[1].head(10)

'''