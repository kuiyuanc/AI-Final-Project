'''
    Run this file directly to give this project a quick try.
'''

import os
import csv
from model import LSTM
from pre_processor import pre_processor


class anime_review_rater:
    """docstring for anime_review_rater"""

    def __init__(self):
        self.animes = []
        self.reviews = []
        self.models = {}

    def load(self, dataset='new'):
        print('load csv...')
        self.load_reviews(dataset)
        self.load_animes()

        # print('load models...')
        # self.load_models()

    def load_reviews(self, dataset):
        with open(f'data/review {dataset}.csv', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for name, text, rating in reader:
                if name == 'Anime':
                    continue
                rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
                self.reviews.append([name, text, rate])

    def load_animes(self):
        with open(f'data/anime.csv', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for anime, genres, rating in reader:
                if anime == 'Anime':
                    continue
                genres = [genre.replace('\'', '').replace('[', '').replace(']', '').replace(' ', '')
                          for genre in genres.split(',')]
                self.animes.append([anime, genres, float(rating)])

    def load_models(self):
        raise NotImplementedError

        for name in os.listdir('models'):
            category, max_feature, input_len, dataset = name.split('-')
            max_feature = 10**1000 if max_feature == 'all' else int(max_feature)

            self.models[name] = model(category, max_feature, input_len, dataset)

            latest = max(os.listdir(f'models/{name}'))
            self.models[name].load(f'models/{latest}.csv')

    def train(self, model):
        raise NotImplementedError

        print('transpose table...\n')
        names, texts, ratings = zip(*self.reviews)
        ratings = [rating / 10 for rating in ratings]


def pre_process():
    arr = anime_review_rater()
    arr.load()

    names, texts, ratings = zip(*arr.reviews)
    pp = pre_processor()
    pp.load(texts, ratings)
    with open('data/processed review new.md', 'w', encoding="utf-8") as md:
        for i in range(len(pp.lemmatized_docs)):
            md.write(f'# review {i}:\n')
            md.write(''.join([word + ' ' for word in pp.lemmatized_docs[i]] + ['\n']))


if __name__ == '__main__':
    arr = anime_review_rater()
    arr.load('old')
    names, texts, ratings = zip(*arr.reviews)
    ratings = [rating / 10 for rating in ratings]
    m = LSTM('LSTM', 2000, 'avg', 'old')
    m.batch(texts, ratings)
    m.build()
    m.train(f'models/{str(m)}/')
    m.rate('Are you stupid ?')
