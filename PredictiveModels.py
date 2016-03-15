from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import cross_validation, datasets, linear_model
'''
def get_ridge_reg_model(user_matrix, alpha):
    user_cols = user_matrix.shape[1]
    data = user_matrix[:, 1:(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]


    regr = linear_model.RidgeCV(alphas=alpha, cv=5)
    regr.fit(data, target)
    y_pred = regr.predict(data)
    mae = mean_absolute_error(target, y_pred)
    weights = regr.coef_
    return [mae, weights, y_pred]


def get_reg_model(user_matrix):
    user_cols = user_matrix.shape[1]
    data = user_matrix[:, 1:(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.33)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

'''

'''
def get_lasso_cv_results(user_matrix, alpha, movie_id):
    user_cols = user_matrix.shape[1]

    data = user_matrix[:, 1:user_cols-1]
    target = user_matrix[:, user_cols-1]

    movie_index = np.where(user_matrix[:,0] == movie_id)

    x_train = data
    y_train = target

    x_train = np.delete(data, (movie_index), axis = 0)
    y_train = np.delete(target, (movie_index), axis = 0)
    x_test = data[movie_index]
    y_test = target[movie_index]
    lasso = linear_model.Lasso(alpha = alpha)
    lasso.fit(x_train, y_train)

    y_pred = lasso.predict(x_test)

    mae = abs(y_test-y_pred)
    weights = lasso.coef_
    return [mae, weights, y_pred]
'''
'''



def get_lasso_cv_results(user_matrix, alpha):
    alpha = [.1]
    user_cols = user_matrix.shape[1]
    split = int(user_matrix.shape[0]*.66)
    np.random.shuffle(user_matrix)
    data = user_matrix[:, 1:(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    movie_ids = user_matrix[:, 0]
    test_ids = movie_ids[split:]

    X_train = data[:split]
    y_train = target[:split]
    X_test = data[split:]
    y_test = target[split:]

    lasso = linear_model.Lasso()
    lasso.fit(X_train, y_train)
    lasso.alpha = alpha
    y_pred = lasso.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    weights = lasso.coef_
    return [mae, weights, y_pred, y_test, test_ids]

import Classifiers as cs
def get_ridge_cv_results(user_matrix, alpha):
    alpha = 15
    user_cols = user_matrix.shape[1]
    split = int(user_matrix.shape[0]*.66)
    np.random.shuffle(user_matrix)
    data = user_matrix[:, 1:(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    movie_ids = user_matrix[:, 0]
    test_ids = movie_ids[split:]

    X_train = data[:split]
    y_train = target[:split]
    X_test = data[split:]
    y_test = target[split:]

    ridge = linear_model.Ridge()
    ridge.fit(X_train, y_train)
    ridge.alpha = alpha
    y_pred = ridge.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    weights = ridge.coef_

    print ridge.intercept_
    return [mae, weights, y_pred, y_test, test_ids]
'''

'''
def get_lasso_cv_results(user_matrix, alphas):
    user_cols = user_matrix.shape[1]
    #data = user_matrix[:, 22:390]
    #data = user_matrix[:, 1:20]
    #data = user_matrix[:, 392:(user_cols-1)]
    data = user_matrix[:, 1:(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    lasso = linear_model.Lasso()
    all_pred = np.zeros(shape=(0))
    all_indices = np.zeros(shape=(0))
    weights = np.zeros(data.shape[1])
    kf = cross_validation.KFold(user_matrix.shape[0], n_folds=5)
    for alpha in alphas:
        lasso.alpha = alpha
        mae_list = list()
        for train_index, test_index in kf:
            X_train, X_test =data[train_index], data[test_index]
            y_train, y_test = target[train_index], target[test_index]

            lasso.fit(X_train, y_train)
            y_pred = lasso.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)

            all_pred = np.concatenate((all_pred,y_pred))
            all_indices = np.concatenate((all_indices, test_index))

            mae_list.append(mae)
            weights = np.add(weights,lasso.coef_)
    y_pred = np.column_stack((all_indices,all_pred))
    y_pred = y_pred[np.argsort(y_pred[:,0])][:,1]

    mae = np.average(mae_list)

    weights = np.divide(weights, 5)

    return [mae, weights, y_pred]
'''


def get_linear_cv_results(user_matrix, folds):
    user_cols = user_matrix.shape[1]
    data = user_matrix[:, :(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    linear = linear_model.LinearRegression()

    kf = cross_validation.KFold(user_matrix.shape[0], n_folds=folds, shuffle = True, random_state = 16834)
    all_pred = np.zeros(shape=(0))
    all_indices = np.zeros(shape=(0))
    mae_list = list()
    for train_index, test_index in kf:
        X_train, X_test =data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        all_pred = np.concatenate((all_pred,y_pred))
        all_indices = np.concatenate((all_indices, test_index))

        mae_list.append(mae)

    y_pred = np.column_stack((all_indices,all_pred))
    y_pred = y_pred[np.argsort(y_pred[:,0])][:,1]

    mae = np.average(mae_list)
    weights = linear.coef_

    return [mae, weights, y_pred]


def get_ridge_cv_results(user_matrix, alpha, folds):
    user_cols = user_matrix.shape[1]
    data = user_matrix[:, :(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    ridge = linear_model.Ridge()

    kf = cross_validation.KFold(user_matrix.shape[0], n_folds=folds, shuffle = True, random_state = 16834)
    all_pred = np.zeros(shape=(0))
    all_indices = np.zeros(shape=(0))
    ridge.alpha = alpha
    mae_list = list()
    for train_index, test_index in kf:
        X_train, X_test =data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        all_pred = np.concatenate((all_pred,y_pred))
        all_indices = np.concatenate((all_indices, test_index))

        mae_list.append(mae)

    y_pred = np.column_stack((all_indices,all_pred))
    y_pred = y_pred[np.argsort(y_pred[:,0])][:,1]

    mae = np.average(mae_list)
    weights = ridge.coef_

    return [mae, weights, y_pred]


def get_lasso_cv_results(user_matrix, alpha, folds):
    user_cols = user_matrix.shape[1]
    data = user_matrix[:, :(user_cols-1)]
    target = user_matrix[:, (user_cols-1)]

    lasso = linear_model.Lasso()

    kf = cross_validation.KFold(user_matrix.shape[0], n_folds=folds, shuffle = True, random_state = 16834)
    all_pred = np.zeros(shape=(0))
    all_indices = np.zeros(shape=(0))
    lasso.alpha = alpha
    mae_list = list()
    for train_index, test_index in kf:
        X_train, X_test =data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        all_pred = np.concatenate((all_pred,y_pred))
        all_indices = np.concatenate((all_indices, test_index))

        mae_list.append(mae)

    y_pred = np.column_stack((all_indices,all_pred))
    y_pred = y_pred[np.argsort(y_pred[:,0])][:,1]

    mae = np.average(mae_list)
    weights = lasso.coef_

    return [mae, weights, y_pred]


'''
import pickle

infile = open("postprocessed-data/all_user_matrix_num_ratings", "r")
all_user_matrix = pickle.load(infile)
user_id = 38
user_matrix = all_user_matrix[user_id-1]

ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)
user_ratings = ratings[user_id-1]
print len(user_ratings[np.nonzero(user_ratings)])

ridge_alphas = [15]
lasso_alphas = [.1]

ridge_mae = list()
lasso_mae = list()

results = get_ridge_cv_results(user_matrix, ridge_alphas)

print results[2]
'''
'''
for user_matrix in all_user_matrix:
    results = get_ridge_cv_results(user_matrix, ridge_alphas)

    ridge_mae.append(results[0])

    results = get_lasso_cv_results(user_matrix, lasso_alphas)

    lasso_mae.append(results[0])


print np.average(ridge_mae)
print np.average(lasso_mae)
'''
