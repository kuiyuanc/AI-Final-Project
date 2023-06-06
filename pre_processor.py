import csv
import nltk
import math
import statistics
import numpy as np

# nltk.download('all')

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class pre_processor:
    """docstring for pre_proccesor"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.__STOP_WORDS = set(stopwords.words("english") + [',', '.', '\'\''])
        self.__PAD = "__PAD"
        self.__UNK = "__UNK"
        self.RANDOM_STATE = 42

    def docs_lemmatize(self, docs, path=None):
        if path:
            with open(path, encoding='utf-8') as f:
                return [line.split() for line in f if line[:8] != '# review']
        else:
            return [[word for word in self.lemmatize(doc) if word not in self.__STOP_WORDS] for doc in docs]

    def make_index(self, docs):
        bag_of_word = nltk.FreqDist([word for doc in docs for word in doc])
        self.word_index = {x[0]: i + 2 for i, x in enumerate(bag_of_word.most_common())}
        self.word_index |= {self.__PAD: 0, self.__UNK: 1}
        return self.word_index

    def split(self, X, y, train_size, test_size, random_state=None):
        random_state = random_state if random_state else self.RANDOM_STATE

        val_test_ratio = (1 - train_size - test_size) / (1 - train_size)

        Xtrain, Xtemp, ytrain, ytemp = train_test_split(X, y, train_size=train_size,
                                                        random_state=random_state)
        Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, train_size=val_test_ratio,
                                                    random_state=random_state)
        return Xtrain, Xval, Xtest, ytrain, yval, ytest

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def sentence_lemmatize(self, sentence):
        words = []
        for w, pos in pos_tag(word_tokenize(sentence.lower())):
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            words.append(self.lemmatizer.lemmatize(w, pos=wordnet_pos))
        return words

    def lemmatize(self, text):
        return [word for sentence in sent_tokenize(text) for word in self.sentence_lemmatize(sentence)]

    def one_hot(self, word, max_index, word_index=None):
        word_index = word_index if word_index else self.word_index

        if word in word_index:
            return word_index[word] if word_index[word] < max_index else word_index[self.__UNK]
        return word_index[self.__UNK]

    def sent_tokenize(self, doc):
        return sent_tokenize(doc)


def pre_process_base():
    reviews = []
    with open(f'data/review.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for name, text, rating in reader:
            if name == 'Anime':
                continue
            rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
            reviews.append([name, text, rate])

    _, texts, _ = zip(*reviews)
    pp = pre_processor()
    pp.load(texts)
    with open(f'bins/processed review base.md', 'w', encoding="utf-8") as md:
        for i in range(len(pp.lemmatized_docs)):
            md.write(f'# review {i}:\n')
            md.write(''.join([word + ' ' for word in pp.lemmatized_docs[i]] + ['\n']))


def pre_process_double_LSTM():
    reviews = []
    with open(f'data/review.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for name, text, rating in reader:
            if name == 'Anime':
                continue
            rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
            reviews.append([name, text, rate])

    _, texts, _ = zip(*reviews)
    texts = [[sentence + '\n' for sentence in sent_tokenize(text)] for text in texts]

    with open(f'bins/processed review double_LSTM.md', 'w', encoding="utf-8") as md:
        for i in range(len(texts)):
            md.write(f'# review {i}:\n')
            md.writelines(texts[i])


def info_base(dataset='new'):
    with open(f'bins/processed review {dataset} base.md', encoding='utf-8') as md:
        lines = [[word for word in line.split()] for line in md.readlines() if line[:8] != '# review']

    y = np.zeros(len(lines))
    pp = pre_processor()
    lines_train, lines_valid, lines_test, _, _, _ = pp.split(lines, y, .8, .1)

    texts = {'train': lines_train, 'valid': lines_valid, 'test': lines_test}

    for set_name, texts in texts.items():
        input_lens = [len(line) for line in texts]
        num_text = len(texts)

        avg = statistics.mean(input_lens)
        stderr = math.sqrt(statistics.variance(input_lens))

        TARGET_RATIO = 0.003

        num_stderr = 0
        cut_ratio = 1
        while cut_ratio > TARGET_RATIO:
            num_stderr += 1
            cut = [length for length in input_lens if length > avg + stderr * num_stderr]
            cut_ratio = len(cut) / num_text

        print(f'number of vocabulary of {dataset} of {set_name}: ', len(nltk.FreqDist([word for text in texts for word in text])))
        print('\n')

        print(f'average input length of {dataset} of {set_name}: ', avg)
        print(f'standard error of input length of {dataset} of {set_name}: ', stderr)
        print(f'need to add {num_stderr} standard error to reduce ratio of input being cut to {cut_ratio}')
        print('\n')

        print(f'max input length of {dataset} of train: ', max(input_lens))
        print('\n\n')


def info_double_LSTM(dataset='new'):
    with open(f'bins/processed review {dataset} double_LSTM.md', encoding='utf-8') as md:
        lines = md.readlines()

    texts = []
    text = []
    for i in range(1, len(lines)):
        if lines[i][:8] == '# review':
            texts.append([line.split() for line in text])
            text = []
        else:
            text.append(lines[i])

    y = np.zeros(len(texts))
    pp = pre_processor()
    texts_train, texts_valid, texts_test, _, _, _ = pp.split(texts, y, .8, .1)

    texts = {'train': texts_train, 'valid': texts_valid, 'test': texts_test}

    for set_name, texts in texts.items():
        sentence_lengths = [len(line) for text in texts for line in text]
        num_sentence = len(sentence_lengths)

        review_lengths = [len(text) for text in texts]
        num_review = len(texts)

        avg_sentence = statistics.mean(sentence_lengths)
        avg_review = statistics.mean(review_lengths)
        stderr_sentence = math.sqrt(statistics.variance(sentence_lengths))
        stderr_review = math.sqrt(statistics.variance(review_lengths))

        TARGET_RATIO = 0.003

        num_stderr_sentence = 0
        cut_ratio_sentence = 1
        while cut_ratio_sentence > TARGET_RATIO:
            num_stderr_sentence += 1
            cut_sentence = [length for length in sentence_lengths if length > avg_sentence + stderr_sentence * num_stderr_sentence]
            cut_ratio_sentence = len(cut_sentence) / num_sentence

        num_stderr_review = 0
        cut_ratio_review = 1
        while cut_ratio_review > TARGET_RATIO:
            num_stderr_review += 1
            cut_review = [length for length in review_lengths if length > avg_review + stderr_review * num_stderr_review]
            cut_ratio_review = len(cut_review) / num_review

        print(f'average sentence length of {dataset} of {set_name}: ', avg_sentence)
        print(f'standard error of sentence length of {dataset} of {set_name}: ', stderr_sentence)
        print(f'need to add {num_stderr_sentence} standard error to reduce ratio of sentence being cut to {cut_ratio_sentence}')
        print('\n')

        print(f'average review length of {dataset} of {set_name}: ', avg_review)
        print(f'standard error of review length of {dataset} of {set_name}: ', stderr_review)
        print(f'need to add {num_stderr_review} standard error to reduce ratio of sentence being cut to {cut_ratio_review}')
        print('\n')

        print(f'max sentence length of {dataset} of {set_name}: ', max(sentence_lengths))
        print(f'max review length of {dataset} of {set_name}: ', max(review_lengths))
        print('\n\n')


def main():
    dataset = 'new'

    # pre_process_double_LSTM(dataset)
    info_base(dataset)
    # info_double_LSTM(dataset)


if __name__ == '__main__':
    main()
