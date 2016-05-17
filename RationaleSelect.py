from Classifiers import TransparentRidge
from Classifiers import TransparentLinearRegression
from Classifiers import TransparentLasso
from LatestPreprocessing import *
import heapq


def get_rationale_weights(data, target, clf = TransparentRidge()):
    clf.fit(data, target)
    weights = clf.coef_
    return weights


def build_rationales(data, target, weights, freq, num_top = 3):
    rows = data.shape[0]
    cols = data.shape[1]
    x = np.multiply(data,weights)
    maxs = {k:0 for k in range(cols)}
    mins = {k:0 for k in range(cols)}
    for i in range(x.shape[0]):
        row = np.array(x[i])
        rating = target[i]
        if rating > 3.5:
            for r in heapq.nlargest(num_top, range(len(row)), key=row.__getitem__):
                if row[r] > 0:
                    maxs[r] +=1
        elif rating < 2.5:
            for r in heapq.nsmallest(num_top, range(len(row)), key=row.__getitem__):
                if row[r] < 0:
                    mins[r] +=1
    thresh = rows*freq
    for key, value in maxs.items():
        if value<thresh:
            maxs.pop(key)
    for key, value in mins.items():
        if value<thresh:
            mins.pop(key)
    return maxs, mins


# Previous unused methods

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


def print_rationales(names, maxs, mins):
    print names[maxs.keys()]
    print names[mins.keys()]


def get_key_names():
    keywords = pd.read_csv("ml-latest/matched_movies", delimiter='\t', usecols = [0,3], names = ['id', 'key'] )
    x = get_popular_key_dict(keywords.key)
    get_popular_key_dict(keywords.key)
    inv_map = {v: k for k, v in x.items()}
    return np.array([inv_map[k] for k in range(len(x))])