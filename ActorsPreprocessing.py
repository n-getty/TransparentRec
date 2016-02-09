import numpy as np
import re


def gen_movie_set():
    movies = np.genfromtxt('postprocessed-data/sorted_movies_popular_keywords', delimiter = '\t', dtype = str, usecols=2)
    movie_set = set()
    for movie in movies:
        movie_set.add(movie)

    output = open("postprocessed-data/movie_set", "w")

    for item in movie_set:
        print>>output, item


def gen_actress_appearance_list():
    actresss_list = open("imdb/actresses.list")
    actresss_list_lines = actresss_list.readlines()[241:]
    #actresss_list_lines = actresss_list.readlines()[4275704:]

    movie_set = open("postprocessed-data/movie_set")
    movie_set = movie_set.read().split('\n')

    actresss_in_grouplens_films = []
    actress_line = ''
    actress = ''
    for line in actresss_list_lines:
        if line.rstrip():
            line = re.sub('\t+', '\t', line)
            line = line.split('\t')
            if line[0] != '':
                if actress_line != '' and actress_line != actress:
                    actresss_in_grouplens_films.append(actress_line)
                actress = line[0]
                actress_line = actress
            if len(line) > 1:
                appearance = line[1]
                index = appearance.index(')')
                appearance = appearance[:index+1]
                if appearance in movie_set:
                    actress_line += '\t' + appearance

    output = open("postprocessed-data/actress_in_grouplens_films", "w")
    for actress in actresss_in_grouplens_films:
        print>>output, actress


def count_movies_with_actresss():
    actress_film_list = open("postprocessed-data/actress_in_grouplens_films")
    actress_film_list_lines = actress_film_list.readlines()
    movie_set = set()
    for line in actress_film_list_lines:
        line = line.split('\t')
        for movie in line[1:]:
            movie_set.add(movie.strip())
    print len(movie_set)
    output = open("postprocessed-data/movie_set_actress", "w")
    for actress in movie_set:
        print>>output, actress


def gen_popular_actresss():
    actress_film_list = open("postprocessed-data/actress_in_grouplens_films")
    actress_film_list_lines = actress_film_list.readlines()
    output = open("postprocessed-data/popular_movie_set_actress", "w")
    for line in actress_film_list_lines:
        line_list = line.split('\t')
        if(len(line_list) > 10):
            print>>output, line.strip()


def gen_pop_actor_list():
    popular_actress_film_list = open("postprocessed-data/popular_movie_set_actress")
    popular_actress_film_list_lines = popular_actress_film_list.readlines()

    popular_actors_film_list = open("postprocessed-data/popular_movie_set_actors")
    popular_actors_film_list_lines = popular_actors_film_list.readlines()


    pop_actors_list = np.concatenate((popular_actors_film_list_lines,popular_actress_film_list_lines), axis = 0)
    actor_list =[]
    for line in pop_actors_list:
        line = line.split('\t')
        actor_list.append(line[0])

    output = open("postprocessed-data/pop_actor_set", "w")

    for item in actor_list:
        print>>output, item


def gen_actress_matrix():
    popular_actress_film_list = open("postprocessed-data/popular_movie_set_actress")
    popular_actress_film_list_lines = popular_actress_film_list.readlines()

    popular_actors_film_list = open("postprocessed-data/popular_movie_set_actors")
    popular_actors_film_list_lines = popular_actors_film_list.readlines()


    pop_actors_list = np.concatenate((popular_actors_film_list_lines,popular_actress_film_list_lines), axis = 0)

    movies = np.genfromtxt('postprocessed-data/sorted_movies_in_list', delimiter='\t', dtype=None)
    movie_map = {}

    actress_size = len(pop_actors_list)

    for movie in movies:
        movie_map[movie[2]] = movie[0]

    actress_matrix= np.empty((1681, actress_size))

    for x in range(0,actress_size):
        appearances = pop_actors_list[x].split('\t')[1:]
        for appearance in appearances:
            index = movie_map[appearance.strip()]-1
            actress_matrix[index][x] = 1

    np.savetxt("postprocessed-data/actor_and_actress_matrix", actress_matrix, delimiter=",", fmt='%1d')

