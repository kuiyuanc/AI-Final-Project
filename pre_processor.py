import csv
import nltk
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
                return [line.split() for line in f.readlines() if line[:8] != '# review']
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

    def max_doc_len(self):
        return max(len(doc) for doc in self.lemmatized_docs)

    def avg_doc_len(self):
        return sum(len(doc) for doc in self.lemmatized_docs) // len(self.lemmatized_docs)

    def num_vocab(self):
        return len(self.bag_of_word)

    def tf(self, word, doc_index):
        raise NotImplementedError

        doc = ''.join([word + ' ' for word in self.lemmatized_docs[doc_index]])
        return self.text_collection.tf(word, doc)

    def idf(self, word):
        raise NotImplementedError

        return self.text_collection.idf(word)

    def tf_idf(self, word, doc_index):
        raise NotImplementedError

        return self.tf(word, doc_index) * self.idf(word)

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


def pre_process_base(dataset='new'):
    reviews = []
    with open(f'data/review {dataset}.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for name, text, rating in reader:
            if name == 'Anime':
                continue
            rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
            reviews.append([name, text, rate])

    _, texts, _ = zip(*reviews)
    pp = pre_processor()
    pp.load(texts)
    with open(f'bins/processed review {dataset} base.md', 'w', encoding="utf-8") as md:
        for i in range(len(pp.lemmatized_docs)):
            md.write(f'# review {i}:\n')
            md.write(''.join([word + ' ' for word in pp.lemmatized_docs[i]] + ['\n']))


def pre_process_double_LSTM(dataset='new'):
    reviews = []
    with open(f'data/review {dataset}.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for name, text, rating in reader:
            if name == 'Anime':
                continue
            rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
            reviews.append([name, text, rate])

    _, texts, _ = zip(*reviews)
    texts = [[sentence + '\n' for sentence in sent_tokenize(text)] for text in texts]

    with open(f'bins/processed review {dataset} double_LSTM.md', 'w', encoding="utf-8") as md:
        for i in range(len(texts)):
            md.write(f'# review {i}:\n')
            md.writelines(texts[i])


def info_base(dataset='new'):
    with open(f'bins/processed review {dataset} base.md', encoding='utf-8') as md:
        lines = [[word for word in line.split()] for line in md.readlines() if line[:8] != '# review']

    y = np.zeros(len(lines))
    pp = pre_processor()
    lines_train, lines_valid, lines_test, _, _, _ = pp.split(lines, y, .8, .1)

    print(f'number of vocabulary of {dataset} of train: ', len(nltk.FreqDist([word for line in lines_train for word in line])))
    print(f'number of vocabulary of {dataset} of valid: ', len(nltk.FreqDist([word for line in lines_valid for word in line])))
    print(f'number of vocabulary of {dataset} of test: ', len(nltk.FreqDist([word for line in lines_test for word in line])))

    print(f'average input length of {dataset} of train: ', sum(len(line) for line in lines_train) // len(lines_train))
    print(f'average input length of {dataset} of valid: ', sum(len(line) for line in lines_valid) // len(lines_valid))
    print(f'average input length of {dataset} of test: ', sum(len(line) for line in lines_test) // len(lines_test))

    print(f'max input length of {dataset} of train: ', max(len(line) for line in lines_train))
    print(f'max input length of {dataset} of valid: ', max(len(line) for line in lines_valid))
    print(f'max input length of {dataset} of test: ', max(len(line) for line in lines_test))


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

    print(f'average sentence length of {dataset} of train: ', sum(len(line) for text in texts_train for line in text) // sum(len(text) for text in texts_train))
    print(f'average sentence length of {dataset} of valid: ', sum(len(line) for text in texts_valid for line in text) // sum(len(text) for text in texts_valid))
    print(f'average sentence length of {dataset} of test: ', sum(len(line) for text in texts_test for line in text) // sum(len(text) for text in texts_test))

    print(f'average review length of {dataset} of train: ', sum(len(text) for text in texts_train) // len(texts_train))
    print(f'average review length of {dataset} of valid: ', sum(len(text) for text in texts_valid) // len(texts_valid))
    print(f'average review length of {dataset} of test: ', sum(len(text) for text in texts_test) // len(texts_test))

    print(f'max sentence length of {dataset} of train: ', max(len(line) for text in texts_train for line in text))
    print(f'max sentence length of {dataset} of valid: ', max(len(line) for text in texts_valid for line in text))
    print(f'max sentence length of {dataset} of test: ', max(len(line) for text in texts_test for line in text))

    print(f'max review length of {dataset} of train: ', max(len(text) for text in texts_train))
    print(f'max review length of {dataset} of valid: ', max(len(text) for text in texts_valid))
    print(f'max review length of {dataset} of test: ', max(len(text) for text in texts_test))


def main():
    dataset = 'new'

    # pre_process_double_LSTM(dataset)
    info_double_LSTM(dataset)


if __name__ == '__main__':
    main()
