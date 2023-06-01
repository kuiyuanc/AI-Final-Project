'''
    Run this file directly to give this project a quick try.
'''

import os
import csv
import nltk
from model import base, double_LSTM
from pre_processor import pre_processor


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

    def train(self, name, start_epoch=0):
        names, texts, ratings = zip(*self.reviews)
        ratings = [rating / 10 for rating in ratings]
        self.models[name].batch(texts, ratings)

        self.models[name].train(f'models/{name}/')


def pre_process(dataset='new'):
    arr = anime_review_rater()
    arr.load_reviews(dataset)

    _, texts, _ = zip(*arr.reviews)
    pp = pre_processor()
    pp.load(texts)
    with open(f'bins/processed review {dataset}.md', 'w', encoding="utf-8") as md:
        for i in range(len(pp.lemmatized_docs)):
            md.write(f'# review {i}:\n')
            md.write(''.join([word + ' ' for word in pp.lemmatized_docs[i]] + ['\n']))


def info(dataset='new'):
    with open(f'bins/processed review {dataset}.md', encoding='utf-8') as md:
        lines = [[word for word in line.split()] for line in md.readlines() if line[:8] != '# review']

    bag_of_word = nltk.FreqDist([word for line in lines for word in line])

    print(f'number of vocabulary of {dataset}: ', len(bag_of_word))

    print(f'average input length of {dataset}: ', sum(len(line) for line in lines) // len(lines))

    print(f'max input length of {dataset}: ', max(len(line) for line in lines))


def train(category, max_feature, input_len, dataset, start_epoch=0):
    name = category + '-' + max_feature + '-' + input_len + '-' + dataset

    arr = anime_review_rater()
    arr.load_reviews(dataset)

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

    arr.train(name, start_epoch)


def main():
    info('new')

    category = 'base'
    max_feature = '2k'
    input_len = 'avg'
    dataset = 'old'
    start_epoch = 0

    # train(category, max_feature, input_len, dataset, start_epoch)


if __name__ == '__main__':
    main()
