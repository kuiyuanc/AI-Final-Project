'''
    Run this file directly to give this project a quick try.
'''

import os
import csv
from model import base, double_LSTM
# from pre_processor import info, pre_process


class anime_review_rater:
    """docstring for anime_review_rater"""

    def __init__(self):
        self.animes = []
        self.reviews = []
        self.models = {}

    def load_reviews(self, dataset):
        self.reviews = []
        with open(f'data/review {dataset}.csv', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for name, text, rating in reader:
                if name == 'Anime':
                    continue
                rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
                self.reviews.append([name, text, rate])

    def load_animes(self):
        self.animes = []
        with open(f'data/anime.csv', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for anime, genres, rating in reader:
                if anime == 'Anime':
                    continue
                genres = [genre.replace('\'', '').replace('[', '').replace(']', '').replace(' ', '')
                          for genre in genres.split(',')]
                self.animes.append([anime, genres, float(rating)])

    def load_model(self, name, epoch):
        category, _, _, _ = name.split('-')

        if category == 'base':
            self.models[name] = base(*name.split('-'))
        elif category == 'double_LSTM':
            self.models[name] = double_LSTM(*name.split('-'))

        self.models[name].load(f'models/{name}/', epoch)

    def build(self, category, max_feature, input_len, dataset):
        name = category + '-' + max_feature + '-' + input_len + '-' + dataset

        if category == 'base':
            self.models[name] = base(category, max_feature, input_len, dataset)
        elif category == 'double_LSTM':
            self.models[name] = double_LSTM(category, max_feature, input_len, dataset)

        self.models[name].build()

    def train(self, name, start_epoch=0, end_epoch=10):
        _, texts, ratings = zip(*self.reviews)
        ratings = [rating / 10 for rating in ratings]
        self.models[name].batch(texts, ratings, True)

        self.models[name].train(f'models/{name}/', end_epoch)

    def rate(self, name, review):
        _, texts, ratings = zip(*self.reviews)
        ratings = [rating / 10 for rating in ratings]
        self.models[name].batch(texts, ratings, True)

        return self.models[name].rate(review)


def train(category, max_feature, input_len, dataset, start_epoch=0, end_epoch=10):
    name = category + '-' + max_feature + '-' + input_len + '-' + dataset

    print('review loading...')
    arr = anime_review_rater()
    arr.load_reviews(dataset)

    print('model preparing...')
    if start_epoch:
        if name not in os.listdir('models'):
            raise RuntimeError(f'model {name} does not exist')
        elif name + ' ' + str(start_epoch - 1) + '.keras' not in os.listdir(f'models/{name}'):
            raise RuntimeError(f'model {name} {start_epoch - 1} does not exist')
        arr.load_model(name, start_epoch - 1)
    else:
        if name not in os.listdir('models'):
            os.mkdir(f'models/{name}')
        arr.build(category, max_feature, input_len, dataset)

    print('training...')
    arr.train(name, start_epoch, end_epoch)


def rate(category, max_feature, input_len, dataset, epoch):
    name = category + '-' + max_feature + '-' + input_len + '-' + dataset

    arr = anime_review_rater()
    arr.load_reviews(dataset)
    arr.load_model(name, epoch)

    print('rating...')
    score = arr.rate(name, 'Are you stupid?')
    print(f'score of \'Are you stupid?\': {score}')


def debug(category, max_feature, input_len, dataset):
    pass


def main():
    '''
        category:
            'base': base line
            'double': main approach
        max_feature:
            'all': all vocabulary
            '2k': most common 2000
        input_len:
            'max': max review length
            'avg': average review length
        start_epoch:
            int: training start from which epoch
        end_epoch:
            int: training ends at which epoch
        epoch:
            int: use which epoch to run rate()

        Current assignment of argument is safe for you to try.
        It should run without error.
    '''
    category = 'base'
    max_feature = '2k'
    input_len = 'avg'
    dataset = 'old'
    start_epoch = 0
    end_epoch = 9
    epoch = 9

    train(category, max_feature, input_len, dataset, start_epoch, end_epoch)

    rate(category, max_feature, input_len, dataset, epoch)


if __name__ == '__main__':
    main()
