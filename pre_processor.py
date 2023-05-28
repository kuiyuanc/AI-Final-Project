import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import TextCollection
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


class pre_processor:
    """docstring for pre_proccesor"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.__STOP_WORDS = set(stopwords.words("english") + [',', '.', '\'\''])
        self.__PAD = "__PAD"
        self.__UNK = "__UNK"
        self.RANDOM_STATE = 42

    def load(self, docs, golden_ratings):
        self.lemmatized_docs = [self.lemmatize(doc) for doc in docs]

        self.bag_of_word = nltk.FreqDist([word for doc in self.lemmatized_docs for word in doc
                                          if word not in self.__STOP_WORDS])

        self.golden_ratings = golden_ratings

    def batch(self, agent, max_feature, input_len, train_size, test_size, random_state=None):
        if agent == 'LSTM':
            self.word_index = {x[0]: i + 2 for i, x in enumerate(self.bag_of_word.most_common(max_feature))}
            self.word_index |= {self.__PAD: 0, self.__UNK: 1}

            self.one_hot_docs = [[self.one_hot(word, max_feature) for word in doc]
                                 for doc in self.lemmatized_docs]

            X, y = self.uniform(input_len)

            return self.split(X, y, train_size, test_size)
        elif agent == 'TFIDF':
            raise NotImplementedError

            self.text_collection = TextCollection(self.lemmatized_docs)

            for doc in self.lemmatized_docs:
                for word, freq in nltk.FreqDist(doc).most_common(max_feature):
                    pass

    def single(self, doc, agent, max_feature, input_len):
        if agent == 'LSTM':
            x = np.array([[self.one_hot(word, max_feature) for word in self.lemmatize(doc)]])
            return sequence.pad_sequences(x, maxlen=input_len)  # pre-truncation/padding if too long/short
        elif agent == 'TFIDF':
            raise NotImplementedError

    def uniform(self, input_len):
        X = np.array(self.one_hot_docs, dtype=object)
        X = sequence.pad_sequences(X, maxlen=input_len)
        y = np.array(self.golden_ratings)
        return X, y

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

    def lemmatize(self, text):
        words = []
        for sentence in sent_tokenize(text):
            for w, pos in pos_tag(word_tokenize(sentence.lower())):
                wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
                words.append(self.lemmatizer.lemmatize(w, pos=wordnet_pos))
        return words

    def one_hot(self, word, max_feature):
        return self.word_index[word] if word in self.word_index and self.word_index[word] < max_feature\
            else self.word_index[self.__UNK]
