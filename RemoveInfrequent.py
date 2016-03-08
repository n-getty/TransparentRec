from LatestPreprocessing import *

def get_popular_key_dict(keys):
    keydict = {}
    for key in keys:
        if key in keydict:
            keydict[key] +=1
        else:
            keydict[key] = 1
    key_list = []
    for key, value in keydict.iteritems():
        if value >= 100:
            key_list.append(key)
    keydict = dict(zip(key_list,range(len(key_list))))
    return keydict


def get_popular_actors():
    actors_film_list = open("ml-latest/actors_in_grouplens_films")
    actors_film_list_lines = actors_film_list.readlines()
    actor_list = []
    alt_dict = get_alt_id_dict()

    for line in actors_film_list_lines:
        line_list = line.split('\t')
        if(len(line_list) > 30):
            actor_list.append(line_list)
    for actor in actor_list:
        for x in range(1,len(actor)):
            actor[x] = alt_dict[actor[x].rstrip()]
    return actor_list


def get_keyword_matrix():
    keywords = pd.read_csv("ml-latest/matched_movies", delimiter='\t', usecols = [0,3], names = ['id', 'key'])
    key_dict = get_popular_key_dict(keywords.key)
    id_dict = get_id_row_dict()
    key_matrix = np.zeros((len(id_dict), len(key_dict)))
    for x in range(len(keywords.id)):
        id = keywords.id[x]
        key = keywords.key[x]
        if key in key_dict:
            row = id_dict[id]
            column = key_dict[key]
            key_matrix[row][column] = 1
    return key_matrix


def get_actor_matrix():
    actor_list = get_popular_actors()

    id_dict = get_id_row_dict()
    rows = len(id_dict.items())
    columns = len(actor_list)

    actor_matrix = np.zeros((rows,columns))
    for x in range(len(actor_list)):
        for movie in actor_list[x][1:]:
            row = id_dict[movie]
            actor_matrix[row][x] = 1
    return actor_matrix


def get_genre_matrix():
    movies = pd.read_csv("ml-latest/movies.csv", delimiter=',', quotechar='"')
    id_dict = get_id_row_dict()
    genre_dict = get_genre_dict()
    genre_matrix = np.zeros((len(id_dict),20))
    for x in range(len(movies)):
        id = movies.movieId[x]
        if id in id_dict:
            genres = movies.genres[x].split('|')
            for genre in genres:
                genre_matrix[id_dict[id]][genre_dict[genre]] = 1
    return genre_matrix


def get_freq_user_keys(key_matrix):
    rows = key_matrix.shape[0]
    thresh = rows*.05
    freq = np.sum(key_matrix, axis=0)
    idxs = np.where(freq>thresh)[0]
    x = key_matrix[:,idxs]
    return x


def get_freq_actor_keys(actor_matrix):
    rows = actor_matrix.shape[0]
    thresh = rows *.01
    freq = np.sum(actor_matrix, axis=0)
    idxs = np.where(freq>thresh)[0]
    x = actor_matrix[:,idxs]
    return x


def getRationales(genre_matrix, user_key_matrix, actor_matrix, target):
    clf =TransparentLinearRegression()
    clf.fit(genre_matrix, target)
    genre_weights = clf.coef_

    clf.fit(user_key_matrix, target)
    key_weights = clf.coef_

    clf.fit(actor_matrix, target)
    actor_weights = clf.coef_

    weights = np.hstack((genre_weights, key_weights, actor_weights))

    keywords = pd.read_csv("ml-latest/matched_movies", delimiter='\t', usecols = [0,3], names = ['id', 'key'] )
    x = get_popular_key_dict(keywords.key)
    inv_map = {v: k for k, v in x.items()}
    key_names = [inv_map[k] for k in range(len(x))]
    genre_names = get_genre_dict().keys()
    actor_names = get_actor_names()
    features = np.hstack([genre_names, key_names, actor_names])

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

    print positive_df.head(10)
    print negative_df.head(10)


def getRationales(data, target, key_names, actor_names):
    clf =TransparentLinearRegression()
    clf.fit(data,target)
    weights = clf.coef_

    genre_names = get_genre_dict().keys()

    features = np.hstack([genre_names, key_names, actor_names])

    positive_weight_indices = weights > 0
    negative_weight_indices = weights < 0

    positive_features = np.column_stack((features[positive_weight_indices], weights[positive_weight_indices].astype(float)))
    negative_features = np.column_stack((features[negative_weight_indices], weights[negative_weight_indices].astype(float)))

    positive_df = pd.DataFrame(positive_features, columns = ["Features", "Weights"])
    positive_df = positive_df.sort("Weights", ascending=0)

    negative_df = pd.DataFrame(negative_features, columns = ["Features", "Weights"])
    negative_df = negative_df.convert_objects(convert_numeric=True)

    negative_df = negative_df.sort("Weights", ascending=1)

    print positive_df.head(10)
    print negative_df.head(10)