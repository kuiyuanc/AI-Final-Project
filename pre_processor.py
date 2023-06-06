import csv
import nltk
import statistics
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


SEPARATER = 'a;weoifawbejwa.kegu@$&*@AHFEOW*A#!$)@*&@#*@)\n'
TARGET_RATIO = 0.003

try:
    stopwords.words("english")
except LookupError:
    nltk.download('stopwords')

try:
    sent_tokenize('')
except LookupError:
    nltk.download('punkt')

try:
    pos_tag([''])
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    wordnet.NOUN
except LookupError:
    nltk.download('wordnet')


class pre_processor:
    """docstring for pre_proccesor"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.__STOP_WORDS = set(stopwords.words("english") + [',', '.', '\'\''])
        self.__PAD = "__PAD"
        self.__UNK = "__UNK"
        self.RANDOM_STATE = 42

    def texts_lemmatize(self, texts):
        return [self.text_lemmatize(text) for text in texts]

    def text_lemmatize(self, text):
        return [self.sentence_lemmatize(sentence) for sentence in sent_tokenize(text)]

    def sentence_lemmatize(self, sentence):
        words = []
        for w, pos in pos_tag(word_tokenize(sentence.lower())):
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            words.append(self.lemmatizer.lemmatize(w, pos=wordnet_pos))
        return [word for word in words if word not in self.__STOP_WORDS]

    def make_index(self, texts):
        bag_of_word = nltk.FreqDist([word for text in texts for word in text])
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

    def one_hot(self, word, max_index, word_index=None):
        word_index = word_index if word_index else self.word_index

        if word in word_index:
            return word_index[word] if word_index[word] < max_index else word_index[self.__UNK]
        return word_index[self.__UNK]


def load_reviews():
    reviews = []
    with open(f'data/review.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for name, text, rating in reader:
            rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
            reviews.append([name, text, rate])
    return reviews


def load_processed_reviews():
    with open(f'bins/processed reviews.txt', encoding='utf-8') as f:
        lines = f.readlines()

    texts = []
    text = []
    for i in range(len(lines)):
        if lines[i] == SEPARATER:
            texts.append(text)
            text = []
        else:
            text.append(lines[i].split())

    return texts


def pre_process():
    reviews = load_reviews()

    _, texts, _ = zip(*reviews)
    pp = pre_processor()
    texts = [[pp.sentence_lemmatize(sentence) for sentence in sent_tokenize(text)] for text in texts]

    with open(f'bins/processed reviews.txt', 'w', encoding="utf-8") as f:
        for text in texts:
            f.writelines([''.join([word + ' ' for word in sentence] + ['\n']) for sentence in text])
            f.write(SEPARATER)


def info_base():
    texts = load_processed_reviews()
    texts = [[word for sentence in text for word in sentence] for text in texts]

    pp = pre_processor()
    texts_train, texts_valid, texts_test, _, _, _ = pp.split(texts, np.zeros(len(texts)), .8, .1)

    texts = {'train': texts_train, 'valid': texts_valid, 'test': texts_test}

    for set_name, texts in texts.items():
        input_lens = [len(text) for text in texts]
        num_text = len(texts)

        avg = statistics.mean(input_lens)
        stdev = statistics.stdev(input_lens)

        num_stdev = 0
        cut_ratio = 0.5
        while cut_ratio > TARGET_RATIO:
            num_stdev += 1
            cut = [length for length in input_lens if length > avg + stdev * num_stdev]
            cut_ratio = len(cut) / num_text

        print(f'number of vocabulary of {set_name}: ', len(nltk.FreqDist([word for text in texts for word in text])))
        print('\n')

        print(f'average input length of {set_name}: ', avg)
        print(f'standard deviation of input length of {set_name}: ', stdev)
        print(f'need to add {num_stdev} standard deviation to reduce ratio of input being cut to {cut_ratio}')
        print('\n')

        print(f'max input length of of {set_name}: ', max(input_lens))
        print('\n\n')


def info_double_LSTM():
    texts = load_processed_reviews()

    pp = pre_processor()
    texts_train, texts_valid, texts_test, _, _, _ = pp.split(texts, np.zeros(len(texts)), .8, .1)

    texts = {'train': texts_train, 'valid': texts_valid, 'test': texts_test}

    for set_name, texts in texts.items():
        sentence_lengths = [len(line) for text in texts for line in text]
        num_sentence = len(sentence_lengths)

        review_lengths = [len(text) for text in texts]
        num_review = len(texts)

        avg_sentence = statistics.mean(sentence_lengths)
        avg_review = statistics.mean(review_lengths)
        stdev_sentence = statistics.stdev(sentence_lengths)
        stdev_review = statistics.stdev(review_lengths)

        num_stdev_sentence = 0
        cut_ratio_sentence = 0.5
        while cut_ratio_sentence > TARGET_RATIO:
            num_stdev_sentence += 1
            cut_sentence = [length for length in sentence_lengths
                            if length > avg_sentence + stdev_sentence * num_stdev_sentence]
            cut_ratio_sentence = len(cut_sentence) / num_sentence

        num_stdev_review = 0
        cut_ratio_review = 0.5
        while cut_ratio_review > TARGET_RATIO:
            num_stdev_review += 1
            cut_review = [length for length in review_lengths
                          if length > avg_review + stdev_review * num_stdev_review]
            cut_ratio_review = len(cut_review) / num_review

        print(f'average sentence length of {set_name}: ', avg_sentence)
        print(f'standard deviation of sentence length of {set_name}: ', stdev_sentence)
        print(f'need to add {num_stdev_sentence} standard deviation to reduce ratio of sentence being cut to {cut_ratio_sentence}')
        print('\n')

        print(f'average review length of {set_name}: ', avg_review)
        print(f'standard deviation of review length of {set_name}: ', stdev_review)
        print(f'need to add {num_stdev_review} standard deviation to reduce ratio of sentence being cut to {cut_ratio_review}')
        print('\n')

        print(f'max sentence length of {set_name}: ', max(sentence_lengths))
        print(f'max review length of {set_name}: ', max(review_lengths))
        print('\n\n')


def main():
    pass


if __name__ == '__main__':
    main()
