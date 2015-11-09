import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter




def plot_all_ratings():

    ratings = np.genfromtxt('postprocessed-data/user_ratings', delimiter=',', dtype=None)

    x = ratings.flat
    x = x[np.nonzero(x)]
    plt.hist(x, np.arange(1,7))
    #plt.hist(x, np.arange(1,7),weights=np.zeros_like(x) + 1. / len(x))
    #formatter = FuncFormatter(to_percent)
    #plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Rating Occurrence")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()


def plot_user_ratings(user_id):
    import matplotlib.pyplot as plt

    ratings = np.genfromtxt('postprocessed-data/user_ratings.csv', delimiter=',', dtype=None)

    x = ratings[user_id-1]

    plt.hist(x, np.arange(1,6))
    plt.title("Ratings Histogram")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()


def plot_movie_keywords():
    keyword_matrix = np.genfromtxt('postprocessed-data/keywords_matrix', delimiter=',', dtype=int)

    x = np.sum(keyword_matrix, axis=0)
    plt.hist(x, np.arange(50,275,25))
    #plt.hist(x, np.arange(50,275,25),weights=np.zeros_like(x) + 1. / len(x))
    #formatter = FuncFormatter(to_percent)
    #plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Keyword Occurrence")
    plt.xlabel("Occurrence")
    plt.ylabel("Frequency")
    plt.show()


def plot_movie_genre():
    movie_genres = np.genfromtxt('ml-100k/u.item', delimiter='|', usecols=range(5, 24))

    x = np.sum(movie_genres, axis=0)
    plt.hist(x, np.arange(0,300,25))
    #plt.hist(x, np.arange(0,300,25),weights=np.zeros_like(x) + 1. / len(x))
    #formatter = FuncFormatter(to_percent)
    #plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Genre Occurrence")
    plt.xlabel("Occurrence")
    plt.ylabel("Frequency")
    plt.show()


def to_percent(y, position):
    s = str(100 * y)

    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'


def plot_mae_list(mae_list):
    plt.hist(mae_list, np.arange(0.,1.8,.2))
    #plt.hist(mae_list, np.arange(0.,1.8,.2),weights=np.zeros_like(mae_list) + 1. / len(mae_list))
    #formatter = FuncFormatter(to_percent)
    #plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Ridge MAE")
    plt.xlabel("MAE")
    plt.ylabel("Frequency")
    plt.show()


def gen_keyword_occurance():
    keyword_matrix = np.genfromtxt('postprocessed-data/keywords_matrix', delimiter=',', dtype=int)

    x = np.sum(keyword_matrix, axis=0)

    output = file('postprocessed-data/keyword_occurrence', 'w')

    for val in x:
        print>>output, val


def gen_user_rating_table(user_data, user_id, model_mae, rating_mae, avg_rating):
    colLabels=( "Average Rating", "User Rating", "Model Prediction", "Feature", "Weight")
    nrows, ncols = len(user_data)+1, len(colLabels)
    hcell, wcell = 0.3, 1.
    hpad, wpad = 2, 2
    fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.title("User Ratings and Predictions for User " + str(user_id)
              + '\nModel MAE: %.2f' % model_mae
                + '\nAverage Rating MAE: %.2f' % rating_mae
                  + '\nUser Average Rating: %.2f' % avg_rating)
    table = ax.table(cellText=user_data,
            colLabels=colLabels,
            loc='center')
    plt.savefig('tables/user_' + str(user_id) + '_full_table.png', bbox_inches='tight', bbox_extra_artists=[table])
    plt.close()


def gen_user_coef_table(user_data, user_id, model_mae, rating_mae):
    colLabels=( "Feature", "Weight")
    nrows, ncols = len(user_data)+1, len(colLabels)
    hcell, wcell = 0.3, 1.
    hpad, wpad = 5, 5
    fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.title("Model Weights for User " + str(user_id)
              + '\nModel MAE: %.2f' % model_mae
                + '\nAverage Rating MAE: %.2f' % rating_mae)
    table = ax.table(cellText=user_data,
            colLabels=colLabels,
            loc='center')
    #plt.show()
    plt.savefig('tables/user_' + str(user_id) + '_weight_table.png', bbox_inches='tight', bbox_extra_artists=[table])
    plt.close()

