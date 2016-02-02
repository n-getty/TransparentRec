import numpy as np
import time
import pandas as pd
import re
import json

def get_frequent_movies():
    import collections
    num_ratings = [0]
    movie_ids = np.load("ml-latest-small/ratings.csv.npy")
    print movie_ids
    #movie_ids= movie_ids.tolist()
    counter = collections.Counter(movie_ids)
    counter.values().sort()
    print counter[:20]
    print counter[-20:]
    '''
    id_dict = {}
    for movie_id in movie_ids:
        if movie_id in id_dict:
            id_dict[movie_id] += 1
        else:
            id_dict[movie_id] = 1

    print id_dict
    print id_dict.values()
    counter = collections.Counter(id_dict.values())
    sorted = counter.values().sort()
    print sorted
    '''


def gen_rating_table():
    ratings = np.genfromtxt('ml-latest/ratings.csv', delimiter=',', dtype=None)
    user_ratings = {}
    for rating in ratings:
        print


def isolate_ratings():
    ratings = np.genfromtxt('ml-latest-small/ratings.csv', delimiter=',', dtype=None, usecols = (0,1,2), skip_header=1)
    np.savetxt("ml-latest-small/ratings.csv", ratings, delimiter=",",fmt=['%d','%d','%.1f'])
    np.save("ml-latest-small/ratings.csv", ratings)


def gen_num_ratings():
    ratings = np.load("ml-latest-small/ratings.csv.npy")
    ratings = np.array(ratings.tolist(), dtype=int)
    users = ratings[:,0]
    y = np.bincount(users)
    print y
    print sorted(y)


def gen_user_ratings():
    movie_dict = {}
    movie_count = 0
    user_ratings = np.zeros((669,10325))
    ratings = np.load("ml-latest-small/ratings.csv.npy")
    for x in ratings:
        user_id = x[0]
        movie_id = x[1]
        rating = x[2]
        if movie_id not in movie_dict:
            user_ratings[0][movie_count] = movie_id
            movie_dict[movie_id] = movie_count
            movie_count += 1
        user_ratings[user_id][movie_dict[movie_id]] = rating
    print user_ratings
    np.savetxt("ml-latest-small/user_ratings.csv", user_ratings, delimiter=",", fmt='%.1f')
    np.save("ml-latest-small/user_ratings", user_ratings)


def match_grouplens_with_imdb():
    movies_list = pd.read_csv("ml-latest/movies.csv", delimiter=',', usecols=(0,1), dtype=None, quotechar='"', names = ['ID', 'Title'])
    no_alt = pd.read_csv("ml-latest/movies.csv", delimiter=',', usecols=(0,1), dtype=None, quotechar='"', names = ['ID', 'Title'])
    rows = movies_list.shape[0]
    alt_dict = {}
    for x in range(1,rows):
        title = movies_list['Title'][x]
        start = title.find( '(' )
        end = title.find( ')' )
        alt = title[start+1:end] + title[end+1:]
        alt_dict[alt] = title
        if start != -1 and end != -1 and title[start+1:].find( '(' ) != -1:
            no_alt['Title'][x] = title[:start-1] + title[end+1:]


    imdb_movie_keywords = np.genfromtxt('ml-latest-small/movie_keywords', delimiter = "\t", dtype = str)

    movie_dict = dict(zip(movies_list.Title, movies_list.ID))
    no_alt = dict(zip(no_alt.Title, no_alt.ID))

    match_set = set()

    output = open("ml-latest/matched_movies", "w")
    for line in imdb_movie_keywords:
        title = line[0]
        keyword = line[1]
        alt_title = title.replace('/IX', '')
        alt_title = alt_title.replace('/XII', '')
        alt_title = alt_title.replace('/XIV', '')
        alt_title = alt_title.replace('/X', '')
        alt_title = alt_title.replace('/VIII', '')
        alt_title = alt_title.replace('/VII', '')
        alt_title = alt_title.replace('/VI', '')
        alt_title = alt_title.replace('/IV', '')
        alt_title = alt_title.replace('/V', '')
        alt_title = alt_title.replace('/III', '')
        alt_title = alt_title.replace('/II', '')
        alt_title = alt_title.replace('/I', '')
        if alt_title.startswith('"') and title.endswith('"'):
            alt_title = alt_title[1:-1]
        if alt_title[:4] == 'The ':
            alt_title = alt_title[4:-7] + ', The' + alt_title[-7:]
        elif alt_title[:2] == 'A ':
            alt_title = alt_title[2:-7] + ', A' + alt_title[-7:]

        if alt_title in no_alt:
            match_set.add(no_alt[alt_title])
            movie = str(no_alt[alt_title]) + '\t' + alt_title + '\t' + title + '\t' + keyword
            print>>output, movie

        elif alt_title in alt_dict:
            lens = alt_dict[alt_title]
            id = movie_dict[lens]
            match_set.add(id)
            movie = str(movie_dict[lens]) + '\t' + alt_title + '\t' + title + '\t' + keyword
            print>>output, movie

    print len(match_set)


def remove_unrated():
    user_ratings = np.load("ml-latest-small/user_ratings.npy")
    print user_ratings.shape
    idxs = []
    x = np.delete(user_ratings,0,0)
    for col in range(0,user_ratings.shape[1]):
        if np.sum(x[:,col]) == 0:
            idxs.append(col)
    user_ratings = np.delete(user_ratings, idxs, 1)
    print user_ratings.shape


def gen_unmatched():
    movies = pd.read_csv("ml-latest/movies.csv", delimiter=',', usecols=(0,1), quotechar='"',dtype={'movieId': np.int32, 'title' : np.str_})
    movie_dict = dict(zip(movies.movieId, movies.title))

    movies_list = np.genfromtxt("ml-latest/matched_movies", delimiter='\t', usecols=(0), dtype=int)

    movie_set = set(movies.movieId)
    matched_set = set(movies_list)

    print len(matched_set)
    print len(movie_set)

    unmatched = movie_set-matched_set
    print len(unmatched)

    output = open("ml-latest/unmatched", 'w')
    for id in unmatched:
        print>>output, id,"\t",movie_dict[id]


def gen_imdb_movie_set():
    imdb_movie = np.genfromtxt('ml-latest/movie_keywords', delimiter = "\t", dtype = str, usecols=0)

    imdb_set = set(imdb_movie)

    imdb = np.array(list(imdb_set))

    np.savetxt("ml-latest/imdb_movie_set", imdb, delimiter=",", fmt='%s')


def gen_imdb_movie_keywords():
    movie_keywords_file = open("ml-latest-small/keywords.list")
    movie_keywords_lines = movie_keywords_file.readlines()[80694:]

    output = open("ml-latest-small/movie_keywords", "w")

    for item in movie_keywords_lines:
        year = re.search('(\(\d{4})|(\(\?\?\?\?)', item).group(0)
        if not item[0] == "\"" and not re.search('(\(TV\))|(\(VG\))',item) and not year == "(????" and not '#' in item and not '{' in item and not '}' in item:
            print>>output, re.sub('[\t]+', '\t', item).strip()


def gen_matched_user_ratings():
    start = time.time()
    movies_list = np.genfromtxt("ml-latest/matched_movies", delimiter='\t', usecols=(0), dtype=int)
    movies_list = np.array(movies_list.tolist(), dtype=int)

    print "Loaded movies in: " + str(time.time()-start)

    unique_matched = np.unique(movies_list)
    num_movies = len(unique_matched)
    movie_dict = dict(zip(unique_matched, np.arange(num_movies)))

    users = 246829
    start = time.time()
    ratings = np.load("ml-latest/ratings.npy")
    print "Loaded ratings: " + str(time.time()-start)

    user_ratings = [[] for _ in range(users+1)]

    start = time.time()

    for rating in ratings:
        user_id = rating[0]
        movie_id = rating[1]
        rating = rating[2]
        if movie_id in movie_dict:
            user_ratings[user_id].append([movie_id,rating])

    print "Dumping pickle: " + str(time.time()-start)
    with open("ml-latest/user_ratings.json" , "wb") as output_file:
        json.dump(user_ratings, output_file)


def gen_user_ratings():
    movies_list = np.genfromtxt("ml-latest/matched_movies", delimiter='\t', usecols=(0), dtype=int)
    movies_list = np.array(movies_list.tolist(), dtype=int)

    movie_dict = {}
    movie_count = 0

    user_ratings = np.zeros((246829,movies_list.shape[0]))
    ratings = np.load("ml-latest-small/ratings.csv.npy")
    for x in ratings:
        user_id = x[0]
        movie_id = x[1]
        rating = x[2]
        if movie_id not in movie_dict:
            user_ratings[0][movie_count] = movie_id
            movie_dict[movie_id] = movie_count
            movie_count += 1
        user_ratings[user_id][movie_dict[movie_id]] = rating
    print user_ratings
    np.savetxt("ml-latest-small/user_ratings.csv", user_ratings, delimiter=",", fmt='%.1f')
    np.save("ml-latest-small/user_ratings", user_ratings)


def gen_unmatched_num_ratings():
    unmatched = pd.read_csv("ml-latest/unmatched", delimiter='\t', quotechar='"',dtype={'id': np.int32, 'title' : np.str_}, names = ['id', 'title'])
    ratings = pd.read_csv("ml-latest/ratings.csv", delimiter=',', quotechar='"',dtype={'userid': np.int32, 'movieid' : np.int32,'rating' : np.float_}, names = ['userid', 'movieid','rating'])

    num_rating_dict = dict.fromkeys(unmatched.id, 0)
    for id in ratings.movieid:
        if id in num_rating_dict:
            num_rating_dict[id] +=1
    unmatched_df = pd.DataFrame(num_rating_dict.items())
    unmatched_df.columns = ['ID', 'Num']
    sorted_unmatched = unmatched_df.sort(['Num'])
    print len(unmatched_df[unmatched_df.Num == 0])
    print sorted_unmatched.tail(20)


def get_keyword_dict():
    movies_list = pd.read_csv("ml-latest/matched_movies", delimiter='\t', quotechar='"',dtype={'id': np.int32, 'title' : np.str_, 'alt' : np.str_, 'key' : np.str_}, names = ['id', 'title', 'alt', 'key'])
    keywords = movies_list.key
    keydict = {}
    for keyword in keywords:
        if keyword in keydict:
            keydict[keyword] += 1
        else:
            keydict[keyword] = 1

    key_list = []
    for key, value in keydict.iteritems():
        if value >= 20:
            key_list.append(key)
    keydict = dict(zip(key_list,range(len(key_list))))
    return keydict


def get_movie_key_dict():
    keydict = get_keyword_dict()
    movies_list = pd.read_csv("ml-latest/matched_movies", delimiter='\t', quotechar='"',dtype={'id': np.int32, 'title' : np.str_, 'alt' : np.str_, 'key' : np.str_}, names = ['id', 'title', 'alt', 'key'])
    movie_key_dict = {k: [] for k in movies_list.id}
    for movie in zip(movies_list.id,movies_list.key):
        id = movie[0]
        key = movie[1]
        if key in keydict:
            #keyid = keydict[key]
            movie_key_dict[id].append(key)
    return movie_key_dict


def get_key_stats(user_ratings):
    keycount = dict()
    movieids = user_ratings[:,0]
    for movieid in movieids:
        keywords = keydict[int(movieid)]
        for keyword in keywords:
            if keyword in keycount:
                keycount[keyword] += 1
            else:
                keycount[keyword] = 1

    key_df = pd.DataFrame(keycount.items())
    key_df.columns = ['Key', 'Occurence']
    sorted_key = key_df.sort(['Occurence'], ascending = 0)
    return sorted_key


def get_genre_count(user_ratings):
    movieids = user_ratings[:,0]
    genredict = dict()
    for movieid in movieids:

        genres = genre_list[id_dict[movieid]].split('|')
        for genre in genres:
            if genre in genredict:
                genredict[genre] += 1
            else:
                genredict[genre] = 1
    genre_df = pd.DataFrame(genredict.items())
    genre_df.columns = ['Genre', 'Occurence']
    sorted_genre = genre_df.sort(['Occurence'], ascending = 0)
    return sorted_genre


def get_movie_id_dict():
    movies_list = pd.read_csv("ml-latest/movies.csv", delimiter=',', quotechar='"')
    id_list = movies_list.movieId
    id_dict = {}
    for x in range(len(id_list)):
        id_dict[id_list[x]] = x
    return id_dict


def get_user_stats(user_ratings):
    ratings = user_ratings[:,1]
    avg = np.average(ratings)
    var = np.var(ratings)
    num = len(ratings)
    sorted_genre = get_genre_count(user_ratings)
    sorted_key = get_key_stats(user_ratings)
    genre_ratio = len(sorted_genre[(sorted_genre['Occurence'] == 1)])/float(len(sorted_genre.Occurence))
    key_ratio = len((sorted_key[(sorted_key['Occurence'] == 1)]))/float(len(sorted_key.Occurence))
    fitness = 100 * (var + genre_ratio + key_ratio)

    return [num, avg, var, genre_ratio, key_ratio, fitness, sorted_genre, sorted_key]

    '''
    print "Number of ratings: ", num
    print "Average Rating: ", avg
    print "Rating variance: ", var
    print "Percentage genres unique:", genre_ratio
    print "Percentage keywords unique:", key_ratio
    print sorted_genre.head(10)
    print sorted_key.head(10)
    print "Overall Fitness:", fitness
    '''


def gen_num_matched_ratings():
    with open("ml-latest/user_ratings.json", "r") as file:
        user_ratings = json.load(file)

    num_ratings = []
    userids = range(1,len(user_ratings))
    for userid in userids:
        num = len(user_ratings[userid])
        num_ratings.append(num)
    numdf = pd.DataFrame(zip(userids,num_ratings))
    numdf.columns = ['User ID', 'Number of Ratings']
    sorted_genre = numdf.sort(['Number of Ratings'], ascending = 0)
    print sorted_genre.head(10)



with open("ml-latest/user_ratings.json", "r") as file:
        user_ratings = json.load(file)
movies_list = pd.read_csv("ml-latest/movies.csv", delimiter=',', quotechar='"')
keydict = get_movie_key_dict()
id_dict = get_movie_id_dict()
keyname = get_keyword_dict()
genre_list = movies_list.genres


def get_all_user_stats():
    result_list = []
    for userid in range(1,246829+1):
        print userid
        ratings = np.array(user_ratings[userid])
        if ratings != []:
            results = [userid].append(get_user_stats(ratings))
            result_list.append(results)
    result_df = pd.DataFrame(result_list[:,range(0,7)],names = ['User ID', 'Number of Ratings', 'Variance', 'Genre Ratio', 'Key Ratio', 'Fitness'])
    sorted_df = result_df.sort(['Fitness'], ascending = 0)
    return sorted_df
