# coding: utf-8

import time
import tmdbsimple as tmdb
import pickle

tmdb.API_KEY = 'c59645c5447d968a9641b9f4733f0ec9'

def update_movies():
    req = tmdb.Movies()

    res = req.popular()
    pages = res['total_pages']

    all_movies = []
    for page in range(1, pages):
        print('{0} / {1} pages'.format(page, pages), flush=True)
        if page % 15 == 0:
            print('waiting for 7 sec to make server rest', flush=True)
            time.sleep(7)
        res = req.popular(page=page)
        movies = res['results']
        count = len(movies)
        print('downloaded {0} movies on page {1}'.format(count, page), flush=True)
        all_movies.extend(movies)
    print('downloaded {0} movies in total', flush=True)
    file_w = open('./movies.pckl', 'wb')
    pickle.dump(movies, file_w)
    file_w.close()
    print('done')

if __name__ == "__main__":
    update_movies()

