import numpy as np
from time import time
import pandas as pd
import re
import json
import progressbar as pb
from sklearn import preprocessing as pp
from pandas import DataFrame as df


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
    movies_list['Title'] = movies_list['Title'].str.upper()
    movie_dict = dict(zip(movies_list.Title, movies_list.ID))

    no_alt = movies_list
    rows = movies_list.shape[0]
    alt_dict = {}
    for x in range(1,rows):
        title = movies_list['Title'][x]
        start = title.find( '(' )
        end = title.find( ')' )
        alt = title[start+1:end] + title[end+1:]
        alt_dict[alt] = title
        if start != -1 and end != -1 and title[start+1:].find( '(' ) != -1:
            alt = title[:start-1] + title[end+1:]
            no_alt['Title'][x] = alt

    imdb_movie_keywords = np.genfromtxt('ml-latest/movie_keywords', delimiter = "\t", dtype = str)

    no_alt = dict(zip(no_alt.Title, no_alt.ID))

    match_set = set()
    output = open("ml-latest/matched_movies", "w")
    previd = -1
    prevtitle = imdb_movie_keywords[0][0]
    temp = prevtitle
    for line in imdb_movie_keywords:
        title = line[0].upper()
        keyword = line[1]
        alt_title = title.replace(' (V)', "")
        if alt_title.startswith('"') and title.endswith('"'):
            alt_title = alt_title[1:-1]
        if alt_title[:4] == 'THE ':
            alt_title = alt_title[4:-7] + ', THE' + alt_title[-7:]
        elif alt_title[:2] == 'A ':
            alt_title = alt_title[2:-7] + ', A' + alt_title[-7:]
        elif alt_title[:3] == 'LA ':
            alt_title = alt_title[3:-7] + ', LA' + alt_title[-7:]
        elif alt_title[:3] == 'EL ':
            alt_title = alt_title[3:-7] + ', EL' + alt_title[-7:]
        elif alt_title[:3] == 'IL ':
            alt_title = alt_title[3:-7] + ', IL' + alt_title[-7:]

        if alt_title in no_alt:
            id = no_alt[alt_title]
            if id != previd:
                match_set.add(previd)
                prevtitle = title
            elif title != prevtitle:
                prevtitle = title
                match_set.add(previd)
            if id not in match_set:
                previd = id
                movie = str(id) + '\t' + alt_title + '\t' + title + '\t' + keyword
                print>>output, movie
        elif alt_title in alt_dict:
            lens = alt_dict[alt_title]
            id = movie_dict[lens]
            if id != previd:
                match_set.add(previd)
                prevtitle = title
            elif title != prevtitle:
                prevtitle = title
                match_set.add(previd)
            if id not in match_set:
                previd = id
                movie = str(id) + '\t' + alt_title + '\t' + title + '\t' + keyword
                print>>output, movie
        elif title in no_alt:
            id = no_alt[title]
            if id != previd:
                match_set.add(previd)
                prevtitle = title
            elif title != prevtitle:
                prevtitle = title
                match_set.add(previd)
            if id not in match_set:
                previd = id
                movie = str(id) + '\t' + title + '\t' + title + '\t' + keyword
                print>>output, movie
    #print len(match_set)


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

    #print len(matched_set)
    #print len(movie_set)

    unmatched = movie_set-matched_set
    print "Number of unmatched movies:", len(unmatched)

    output = open("ml-latest/unmatched", 'w')
    for id in unmatched:
        print>>output, id,"\t",movie_dict[id]


def gen_imdb_movie_set():
    imdb_movie = np.genfromtxt('ml-latest/movie_keywords', delimiter = "\t", dtype = str, usecols=0)

    imdb_set = set(imdb_movie)

    imdb = np.array(list(imdb_set))

    np.savetxt("ml-latest/imdb_movie_set", imdb, delimiter=",", fmt='%s')


def gen_imdb_movie_keywords():
    movie_keywords_file = open("ml-latest/keywords.list")
    movie_keywords_lines = movie_keywords_file.readlines()[80694:]

    output = open("ml-latest/movie_keywords", "w")

    for item in movie_keywords_lines:
        year = re.search('(\(\d{4})|(\(\?\?\?\?)', item).group(0)
        if not item[0] == "\"" and not re.search('(\(TV\))|(\(VG\))',item) and not year == "(????" and not '#' in item and not '{' in item and not '}' in item:
            print>>output, re.sub('[\t]+', '\t', item).strip()


def gen_matched_user_ratings():
    start = time()
    movies_list = np.genfromtxt("ml-latest/matched_movies", delimiter='\t', usecols=(0), dtype=int)
    movies_list = np.array(movies_list.tolist(), dtype=int)

    print "Loaded movies in: " + str(time()-start)

    unique_matched = np.unique(movies_list)
    num_movies = len(unique_matched)
    movie_dict = dict(zip(unique_matched, np.arange(num_movies)))

    users = 246829
    start = time()
    ratings = np.load("ml-latest/ratings.npy")
    print "Loaded ratings: " + str(time()-start)

    user_ratings = [[] for _ in range(users+1)]

    start = time()

    for rating in ratings:
        user_id = rating[0]
        movie_id = rating[1]
        rating = rating[2]
        if movie_id in movie_dict:
            user_ratings[user_id].append([movie_id,rating])

    print "Dumping pickle: " + str(time()-start)
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
    unmatched_df.columns = ['ID', 'Number of Ratings']
    sorted_unmatched = unmatched_df.sort(['Number of Ratings'], ascending =0)
    print len(unmatched_df[unmatched_df.Num == 0])
    print sorted_unmatched.head(10)


def get_movie_key_dict():
    movies_list = pd.read_csv("ml-latest/matched_movies", delimiter='\t', quotechar='"',dtype={'id': np.int32, 'title' : np.str_, 'alt' : np.str_, 'key' : np.str_}, names = ['id', 'title', 'alt', 'key'])
    keydict = get_popular_key_dict(movies_list.key)
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
    fitness = (var/10 - genre_ratio - key_ratio)
    res = [num, avg, var, genre_ratio, key_ratio, fitness]
    return res

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


def get_all_user_stats():
    print "Loaded data"
    users = 246829

    widgets = ['Test: ', pb.Percentage(), ' ', pb.Bar(marker='0',left='[',right=']'),
           ' ', pb.ETA(), ' ', pb.FileTransferSpeed()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=users+1)
    pbar.start()

    result_list = []
    for userid in range(1,users+1):
        ratings = np.array(user_ratings[userid])
        if ratings != []:
            results = [userid] + (get_user_stats(ratings))
            result_list.append(results)
        pbar.update(userid)

    pbar.finish()
    result_df = pd.DataFrame(result_list, columns = ['User ID', 'Number of Ratings', 'Average Rating', 'Variance', 'Genre Ratio', 'Key Ratio', 'Fitness'])
    sorted_df = result_df.sort(['Fitness'], ascending = 0)
    sorted_df.to_csv('ml-latest/user_stats.csv', sep = ',', index = 0, float_format='%.3f')
    return sorted_df


def gen_actor_appearance_list():
    movie_set = set(pd.read_csv("ml-latest/matched_set.csv", sep = ',', quotechar = '"').title)
    print "Loaded movies"

    actress_list = open("ml-latest/actresses.list")
    actress_list_lines = actress_list.readlines()[243:11113113]
    print "Loaded actresses"

    actress_in_grouplens_films = []
    actress_line = ''
    actress = ''

    app = 0

    print "Searching actresses, elpased time:", time()-start
    for line in actress_list_lines:
        if line.rstrip():
            line = re.sub('\t+', '\t', line)
            line = line.split('\t')
            if line[0] != '':
                if actress_line != '' and actress_line != actress:
                    actress_in_grouplens_films.append(actress_line)
                actress = line[0]
                actress_line = actress
            if len(line) > 1:
                appearance = line[1].upper()
                index = appearance.index(')')
                appearance = appearance[:index+1]
                if appearance in movie_set:
                    actress_line += '\t' + appearance
                    app+=1
    actor_list = open("ml-latest/actors.list")
    actor_list_lines = actor_list.readlines()[241:18639753]
    print "Loaded actors"

    actor_in_grouplens_films = []
    actor_line = ''
    actor = ''

    print "Searching actors, elpased time:", time()-start
    for line in actor_list_lines:
        if line.rstrip():
            line = re.sub('\t+', '\t', line)
            line = line.split('\t')
            if line[0] != '':
                if actor_line != '' and actor_line != actor:
                    actor_in_grouplens_films.append(actor_line)
                actor = line[0]
                actor_line = actor
            if len(line) > 1:
                appearance = line[1].upper()
                index = appearance.index(')')
                appearance = appearance[:index+1]
                if appearance in movie_set:
                    actor_line += '\t' + appearance
                    app+=1
    print "Writing file, elpased time:", time()-start
    print "Total matched appearances:", app
    output = open("ml-latest/actors_in_grouplens_films", "w")
    for actress in actress_in_grouplens_films:
        print>>output, actress

    for actor in actor_in_grouplens_films:
        print>>output, actor


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


def get_actor_names():
    pop = get_popular_actors()
    names = []
    for name in pop:
        names.append(name[0])
    return names


def get_alt_id_dict():
    movies = pd.read_csv("ml-latest/matched_set.csv", sep = ",")
    alt_dict = movies.set_index('title')['id'].to_dict()
    return alt_dict


def get_movie_names():
    id_dict = get_id_row_dict()
    movie_set = pd.read_csv("ml-latest/matched_set.csv", sep = ",")
    names = np.empty(len(id_dict),dtype=object)
    for idx, movie in movie_set.iterrows():
        if movie.id in id_dict:
            names[id_dict[movie.id]] = movie.title
    return names


def count_alt():
    movies = pd.read_csv("ml-latest/matched_set.csv", sep = ",")
    count = 0
    for x in range(len(movies.id)):
        if movies.title[x] != movies.alt[x]:
            count+=1
    print count


def gen_matched_set():
    keywords = pd.read_csv("ml-latest/matched_movies", delimiter='\t')
    keywords.columns = ['a', 'b', 'c', 'd']
    id_set = set()

    for x in range(len(keywords.a)):
        id = keywords.a[x]
        title = keywords.b[x]
        alt = keywords.c[x]
        id_set.add((id,title, alt))

    iddf = pd.DataFrame(list(id_set), columns = ['id', 'alt', 'title'])
    iddf.to_csv("ml-latest/matched_set.csv", set = ",")


def get_actor_dict(actor_list):
    actor_dict = {}
    for x in range(len(actor_list)):
        actor_dict[x] = actor_list[x][0]


def get_id_row_dict():
    movie_set = pd.read_csv("ml-latest/matched_set.csv", sep = ",")
    id_dict = {}
    for x in range(len(movie_set.id)):
        if not movie_set.id[x] in id_dict:
            id_dict[movie_set.id[x]] = x
        else:
            print movie_set.id[x]
    #print "Duplicate films:", (len(movie_set.id) - len(id_dict))
    return id_dict


def re_match_movies():
    #gen_imdb_movie_keywords()
    #match_grouplens_with_imdb()
    print 'Matched Movies'
    gen_unmatched()
    print 'Saved unmatched'
    gen_matched_user_ratings()
    print 'Saved matched user ratings'
    gen_matched_set()
    print 'Saved matched movie set'
    gen_actor_appearance_list()
    print 'Saved actor appearance list'
    get_all_user_stats()
    print 'Saved user stats'
    print 'Non unique ids:'
    get_id_row_dict()


def get_popular_key_dict(keys):
    keydict = {}
    for key in keys:
        if key in keydict:
            keydict[key] +=1
        else:
            keydict[key] = 1
    #print len(keydict)
    key_list = []
    for key, value in keydict.iteritems():
        if value >= 100:
            key_list.append(key)
    keydict = dict(zip(key_list,range(len(key_list))))
    return keydict


def get_genre_dict():
    genre_dict = {'Action' : 0
                ,'Adventure' : 1
                ,'Animation' : 2
                ,'Children' : 3
                ,'Comedy' : 4
                ,'Crime' : 5
                ,'Documentary' : 6
                ,'Drama' : 7
                ,'Fantasy' : 8
                ,'Film-Noir' : 9
                ,'Horror' : 10
                ,'Musical' : 11
                ,'Mystery' : 12
                ,'Romance' : 13
                ,'Sci-Fi' : 14
                ,'Thriller' : 15
                ,'War' : 16
                ,'Western' : 17
                ,'IMAX' : 18
                ,'(no genres listed)' : 19}
    return genre_dict


def gen_keyword_matrix():
    keywords = pd.read_csv("ml-latest/matched_movies", delimiter='\t', usecols = [0,3], names = ['id', 'key'])
    key_dict = get_popular_key_dict(keywords.key)
    #print "Number of popular keys:", len(key_dict)
    id_dict = get_id_row_dict()
    key_matrix = np.zeros((len(id_dict), len(key_dict)))
    for x in range(len(keywords.id)):
        id = keywords.id[x]
        key = keywords.key[x]
        if key in key_dict:
            row = id_dict[id]
            column = key_dict[key]
            key_matrix[row][column] = 1
    np.savetxt("ml-latest/key_matrix.csv", key_matrix, delimiter=",", fmt = '%1i')


def gen_actor_matrix():
    actor_list = get_popular_actors()

    id_dict = get_id_row_dict()
    rows = len(id_dict.items())
    columns = len(actor_list)

    actor_matrix = np.zeros((rows,columns))
    for x in range(len(actor_list)):
        for movie in actor_list[x][1:]:
            row = id_dict[movie]
            actor_matrix[row][x] = 1
    np.savetxt("ml-latest/actor_matrix.csv", actor_matrix, delimiter=",", fmt = '%1i')


def gen_genre_matrix():
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
    np.savetxt("ml-latest/genre_matrix.csv", genre_matrix, delimiter=",", fmt = '%1i')


def get_actor_matrix():
    actor_matrix = np.loadtxt("ml-latest/actor_matrix.csv", delimiter=",")
    return actor_matrix


def get_key_matrix():
    key_matrix = np.loadtxt("ml-latest/key_matrix.csv", delimiter=",")
    return key_matrix


def get_genre_matrix():
    genre_matrix = np.loadtxt("ml-latest/genre_matrix.csv", delimiter=",")
    return genre_matrix


def get_matched_user_ratings(userid):
    id_dict = get_id_row_dict()

    ratings = pd.read_csv("ml-latest/ratings.csv", delimiter=',', quotechar='"', names = ['userId', 'movieId', 'rating'])

    user_ratings = ratings.loc[ratings['userId'] == userid]

    rating_dict = {}

    for index, row in user_ratings.iterrows():
        movie_id = row['movieId']
        rating = row['rating']
        if movie_id in id_dict:
            rating_dict[movie_id] = rating

    return rating_dict


def get_average_and_num_ratings():
    ratings = pd.read_csv("ml-latest/ratings.csv", delimiter=',', quotechar='"', usecols = [1,2], names = ['movieId', 'rating'])
    id_dict = get_id_row_dict()
    ratings = ratings.loc[ratings['movieId'].isin(id_dict.keys())]
    avg = ratings.groupby('movieId').agg(['mean', 'count'])
    #avg.to_csv('ml-latest/avg_and_num', sep = ',', fmt=['%.2f', '%1i'])
    avg_and_num = np.empty((len(id_dict),2))
    for index, row in avg.iterrows():
        idx = id_dict[index]
        print row.rating.mean.count
        exit(0)
        avg_and_num[idx][0] = row['mean']
        avg_and_num[idx][1] = row['count']
    return avg_and_num


def sort_avg():
    avg = pd.read_csv('ml-latest/avg_and_num', sep = ',')
    id_dict = get_id_row_dict()
    avg_and_num = np.empty((len(id_dict),2))
    for index, row in avg.iterrows():
        idx = id_dict[row.movieId]
        avg_and_num[idx][0] = row['mean']
        avg_and_num[idx][1] = row['count']
    return avg_and_num



def get_sub_avg(movies, avg):
    return avg


def gen_user_results_old(user_ratings, results, avg_ratings):
    y_pred = results[2]
    ratings = user_ratings.values()
    error_from_average = np.abs(avg_ratings-ratings)
    error_from_pred = np.abs(ratings-y_pred)

    col_names = ["Movie ID" ,"User Rating","Model Prediction","Model Error"]

    col_avg = ["Average:", np.average(ratings),np.average(y_pred),np.average(error_from_pred)]
    user_result = np.column_stack((user_ratings.keys(), ratings, y_pred, error_from_pred))
    user_result = user_result[np.argsort(user_result[:, 3])]
    user_result = np.vstack((user_result,col_avg))

    df = pd.DataFrame(user_result, columns = col_names)
    #df = df.convert_objects(convert_numeric=True)

    return df

'''
with open("ml-latest/user_ratings.json", "r") as file:
        user_ratings = json.load(file)
movies_list = pd.read_csv("ml-latest/movies.csv", delimiter=',', quotechar='"')
matched = pd.read_csv("ml-latest/matched_movies", delimiter='\t', quotechar='"',dtype={'id': np.int32, 'title' : np.str_, 'alt' : np.str_, 'key' : np.str_}, names = ['id', 'title', 'alt', 'key'])
keydict = get_movie_key_dict()
id_dict = get_movie_id_dict()
keyname = get_popular_key_dict(matched.key)
genre_list = movies_list.genres
#get_all_user_stats()
userid = 28451
ratings = np.array(user_ratings[userid])
print get_user_stats(ratings)
userid = 170099
ratings = np.array(user_ratings[userid])
print get_user_stats(ratings)
'''