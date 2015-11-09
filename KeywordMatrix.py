import numpy as np


# List of keywords in group lens films that appear a given number of times
def get_common_keywords_list(min):
    keywords = np.genfromtxt('postprocessed-data/sorted_movies_in_list', delimiter='\t', dtype=None, usecols=3)

    keydict = {}

    for keyword in keywords:
        if keyword in keydict:
            keydict[keyword] += 1
        else:
            keydict[keyword] = 1

    key_list = []

    for key, value in keydict.iteritems():
        if value > min-1:
            key_list.append(key)

    return key_list


# Create list of all films and their popular keywords
def get_popular_movies_list(keywords):
    movies = np.genfromtxt('postprocessed-data/sorted_movies_in_list', delimiter='\t', dtype=None)
    keywords = set(keywords)

    newarr = np.empty((0, 4))

    for movie in movies:
        if movie[3] in keywords:
            moviearr = np.array([[movie[0], movie[1], movie[2], movie[3]]])
            newarr = np.append(newarr, moviearr, axis=0)

    return newarr


# Create the movie keyword matrix, each row is a movie with index corresponding to film and each column is a keyword
def gen_keyword_matrix(min):
    keywords = get_common_keywords_list(min)
    movies = get_popular_movies_list(keywords)

    keywords_matrix = np.zeros((1681, len(keywords)))
    for item in movies:
        id = int(item[0]) - 1
        keyword = item[3]
        index = keywords.index(keyword)
        keywords_matrix[id][index] = 1

    np.savetxt("postprocessed-data/keywords_matrix", keywords_matrix, delimiter=",", fmt='%1d')
