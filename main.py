'''
    Run this file directly to give this project a quick try.
'''

import os
import csv
from model import base, double_LSTM


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

    def train(self, name, end_epoch=10):
        self.batch(name)
        self.models[name].train(f'models/{name}/', end_epoch)

    def test(self, name):
        self.batch(name)
        self.models[name].test(f'models/{name}/')

    def rate(self, name, review):
        self.batch(name)
        return self.models[name].rate(review)

    def batch(self, name):
        _, texts, ratings = zip(*self.reviews)
        ratings = [rating / 10 for rating in ratings]
        self.models[name].batch(texts, ratings, True)

    def info(self, name):
        self.models[name].info()


def load_arr(category, max_feature, input_len, dataset, epoch, load=True):
    name = category + '-' + max_feature + '-' + input_len + '-' + dataset

    arr = anime_review_rater()
    arr.load_reviews(dataset)

    if load is False:
        return arr

    if name not in os.listdir('models'):
        raise RuntimeError(f'model {name} does not exist')
    elif name + f' {epoch}.keras' not in os.listdir(f'models/{name}'):
        raise RuntimeError(f'model {name} {epoch} does not exist')

    if load:
        arr.load_model(name, epoch)

    return arr


def train(category, max_feature, input_len, dataset, start_epoch=0, end_epoch=10):
    name = category + '-' + max_feature + '-' + input_len + '-' + dataset

    arr = load_arr(category, max_feature, input_len, dataset, start_epoch - 1) if start_epoch\
        else load_arr(category, max_feature, input_len, dataset, None, False)

    if start_epoch == 0:
        if name not in os.listdir('models'):
            os.mkdir(f'models/{name}')
        arr.build(category, max_feature, input_len, dataset)

    arr.train(name, end_epoch)


def main():
    '''
        category:
            'base': base line
            'double_LSTM': main approach
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
    category = 'double_LSTM'
    max_feature = '2k'
    input_len = 'max'
    dataset = 'new'
    start_epoch = 0
    end_epoch = 0
    epoch = 0

    print('training...')
    train(category, max_feature, input_len, dataset, start_epoch, end_epoch)
    print('\n')

    name = category + '-' + max_feature + '-' + input_len + '-' + dataset
    arr = load_arr(category, max_feature, input_len, dataset, epoch)

    print('testing...')
    arr.test(name)
    print('\n')

    arr.info(name)
    print('\n')

    print('rating...')
    doc = 'Are you stupid?'
    score = arr.rate(name, doc)
    print(f'score of "{doc}": {score}')


if __name__ == '__main__':
    main()
