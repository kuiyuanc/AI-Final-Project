'''
    Run this file directly to give this project a quick try.
'''

import os
from model import base, double_LSTM
from pre_processor import load_reviews


class anime_review_rater:
    """docstring for anime_review_rater"""

    def __init__(self):
        self.reviews = []
        self.models = {}

    def load_reviews(self):
        self.reviews = load_reviews()

    def load_model(self, name, epoch):
        category, _, _ = name.split('-')

        if category == 'base':
            self.models[name] = base(*name.split('-'))
        elif category == 'double_LSTM':
            self.models[name] = double_LSTM(*name.split('-'))

        self.models[name].load(f'models/{name}/', epoch)

    def build(self, category, max_feature, input_len):
        name = category + '-' + max_feature + '-' + input_len

        if category == 'base':
            self.models[name] = base(category, max_feature, input_len)
        elif category == 'double_LSTM':
            self.models[name] = double_LSTM(category, max_feature, input_len)

        self.models[name].build()

    def train(self, name, end_epoch=9):
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
        self.models[name].batch(texts, ratings, True if 'processed reviews.txt' in os.listdir('bins') else False)

    def info(self, name):
        self.models[name].info()


def load_arr(category, max_feature, input_len, epoch, load=True):
    name = category + '-' + max_feature + '-' + input_len

    arr = anime_review_rater()
    arr.load_reviews()

    if load is False:
        return arr

    if name not in os.listdir('models'):
        raise RuntimeError(f'model {name} does not exist')
    elif name + f' {epoch}.keras' not in os.listdir(f'models/{name}'):
        raise RuntimeError(f'model {name} {epoch} does not exist')

    arr.load_model(name, epoch)

    return arr


def train(category, max_feature, input_len, start_epoch=0, end_epoch=9):
    name = category + '-' + max_feature + '-' + input_len

    if start_epoch:
        arr = load_arr(category, max_feature, input_len, start_epoch - 1)
    else:
        arr.build(category, max_feature, input_len)

    if name not in os.listdir('models'):
        os.mkdir(f'models/{name}')

    arr.train(name, end_epoch)


def main():
    '''
        category:
            'base': baseline
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
            int: use which epoch to run arr.test(), arr.info(), and arr.rate()

        Current assignment of argument is safe for you to try.
        It should run without error.
    '''
    category = 'base'
    max_feature = '2k'
    input_len = 'avg'
    start_epoch = 0
    end_epoch = 0
    epoch = 0

    print('training...')
    train(category, max_feature, input_len, start_epoch, end_epoch)
    print('\n')

    name = category + '-' + max_feature + '-' + input_len
    arr = load_arr(category, max_feature, input_len, epoch)

    print('testing...')
    arr.test(name)
    print('\n')

    arr.info(name)
    print('\n')

    print('rating...')
    text = 'Any text you want to analyze.'
    score = arr.rate(name, text)
    print(f'score of "{text}": {score}')


if __name__ == '__main__':
    main()
