from Classifiers import TransparentLinearRegression
from LatestPreprocessing import *


def get_global_weights(weights, data):
    freq = np.sum(data, axis=0)
    return np.multiply(freq, weights)


def get_rationale_weight(data, target):
    clf =TransparentLinearRegression()
    clf.fit(data,target)
    weights = clf.coef_
    alt_weights = get_global_weights(weights, data)
    return weights, alt_weights


def get_rationales(weights, movie):
    movie = movie[2:]
    weights = weights[0]
    alt_weights = weights[1]

    likelihood = np.multiply(movie, weights)
    alt_likelihood = np.multiply(movie,alt_weights)

    top = likelihood.argsort()[::-1]
    alt_top = alt_likelihood.argsort()[-3:][::-1]

    top = np.hstack((top[:3], top[-3:]))
    alt_top = np.hstack((alt_top[:3], alt_top[-3:]))

    return top, alt_top


def get_freq_user_keys(key_matrix):
    keywords = pd.read_csv("ml-latest/matched_movies", delimiter='\t', usecols = [0,3], names = ['id', 'key'] )
    x = get_popular_key_dict(keywords.key)
    inv_map = {v: k for k, v in x.items()}

    rows = key_matrix.shape[0]
    thresh = rows*.05
    freq = np.sum(key_matrix, axis=0)
    idxs = np.where(freq>thresh)[0]
    key_names = [inv_map[k] for k in range(len(x))]
    key_names = [key_names[i] for i in idxs]
    x = key_matrix[:,idxs]
    return x, key_names


def get_freq_actor(actor_matrix):
    rows = actor_matrix.shape[0]
    thresh = rows *.05
    freq = np.sum(actor_matrix, axis=0)
    idxs = np.where(freq>thresh)[0]
    actor_names = get_actor_names()
    actor_names = [actor_names[i] for i in idxs]
    x = actor_matrix[:,idxs]
    return x, actor_names