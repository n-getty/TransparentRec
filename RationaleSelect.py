from Classifiers import TransparentLinearRegression
from LatestPreprocessing import *


def get_rationale_weights(data, target):
    clf =TransparentLinearRegression()
    clf.fit(data, target)
    weights = clf.coef_
    return weights


def build_rationales(data, target, weights, freq):
    rows = data.shape[0]
    cols = data.shape[1]
    x = np.multiply(data,weights)
    maxs = {k:0 for k in range(cols)}
    mins = {k:0 for k in range(cols)}
    for i in range(x.shape[0]):
        row = x[i]
        rating = target[i]
        if rating > 3.0:
            for r in np.argpartition(row, -3)[-3:]:
                maxs[r] +=1
        elif rating < 3.0:
            for r in np.argpartition(row, -3)[:3]:
                mins[r] +=1
    thresh = rows*freq
    for key, value in maxs.items():
        if value<thresh:
            maxs.pop(key)
    for key, value in mins.items():
        if value<thresh:
            mins.pop(key)
    return maxs, mins


#Previous methods


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


'''
#userid = 28451
userid = 242763
user_ratings = get_matched_user_ratings(userid)
movies = user_ratings.keys()
ratings = user_ratings.values()

id_dict = get_id_row_dict()
idxs = [id_dict[k] for k in movies]

genre_matrix = get_genre_matrix()[idxs]
#key_matrix  = get_key_matrix()[idxs]
#actor_matrix = get_actor_matrix()[idxs]

#key_names = get_key_names()
genre_names = np.array(get_genre_dict().keys())
#actor_names = np.array(get_actor_names())


rationale_weights = get_rationale_weights(genre_matrix, ratings)
maxs, mins = build_rationales(genre_matrix, rationale_weights, 0.1)
print_rationales(genre_names, maxs, mins)
'''

'''
rationale_weights = get_rationale_weights(key_matrix, ratings)
maxs, mins = build_rationales(key_matrix, rationale_weights, 0.05)
print_rationales(key_names, maxs, mins)

rationale_weights = get_rationale_weights(actor_matrix, ratings)
maxs, mins = build_rationales(actor_matrix, rationale_weights, 0.01)
print_rationales(actor_names, maxs, mins)


rationale_data = np.column_stack((genre_matrix, key_matrix, actor_matrix))
rationale_weights = get_rationale_weights(rationale_data, ratings)
maxs, mins = build_rationales(rationale_data, rationale_weights, 0.01)
names = np.hstack([genre_names, key_names, actor_names])
print_rationales(names, maxs, mins)
'''
