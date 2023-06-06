import json
import numpy as np
import tensorflow as tf
from pre_processor import pre_processor
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras import Model


NUM_VOCAB_NEW = 108333

INPUT_LENGTH_AVG_NEW = 519
INPUT_LENGTH_MAX_OLD = 2565
# INPUT_LENGTH_MAX_NEW = 7734
INPUT_LENGTH_MAX_NEW = 3594

SENTENCE_LENGTH_AVG_OLD = 18
SENTENCE_LENGTH_AVG_NEW = 17
SENTENCE_LENGTH_MAX_OLD = 298
# SENTENCE_LENGTH_MAX_NEW = 464
# SENTENCE_LENGTH_MAX_NEW = 59
SENTENCE_LENGTH_MAX_NEW = 101
REVIEW_LENGTH_AVG_OLD = 40
REVIEW_LENGTH_AVG_NEW = 25
REVIEW_LENGTH_MAX_OLD = 236
# REVIEW_LENGTH_MAX_NEW = 390
# REVIEW_LENGTH_MAX_NEW = 96
REVIEW_LENGTH_MAX_NEW = 169


class model:
    """docstring for model"""

    def __init__(self, category, max_feature, input_len, dataset):
        self.TRAIN_SIZE = 0.8
        self.TEST_SIZE = 0.1
        self.BATCH_SIZE = 32
        self.VERBOSE = 1

        self.category = category
        self.max_feature = max_feature
        self.input_len = input_len
        self.dataset = dataset
        self.epoch = -1

        self.pre_processor = pre_processor()
        self.model = Sequential(name=str(self))
        self.training_history = []

    def __str__(self):
        return self.category + '-' + self.max_feature + '-' + self.input_len + '-' + self.dataset

    def batch(self, docs, golden_ratings, pre_done=None):
        raise NotImplementedError

    def single(self, doc):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def train(self, path, end_epoch):
        Xtrain, Xval, Xtest, ytrain, yval, ytest = self.pre_processor.split(self.X, self.y,
                                                                            self.TRAIN_SIZE, self.TEST_SIZE)

        for i in range(self.epoch + 1, end_epoch + 1):
            print(f'epoch {i}:')
            history = self.model.fit(Xtrain, ytrain, batch_size=self.BATCH_SIZE, validation_data=(Xval, yval))
            self.training_history.append(history)
            self.epoch += 1
            self.save(path)

    def test(self, path):
        _, _, Xtest, _, _, ytest = self.pre_processor.split(self.X, self.y,
                                                            self.TRAIN_SIZE, self.TEST_SIZE)

        loss = self.model.evaluate(Xtest, ytest, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        with open(path + str(self) + f' {self.epoch} test.json', 'w') as history_file:
            history_file.write(f'{{"loss": [{loss}]}}')

    def load(self, path, epoch):
        self.epoch = epoch
        self.model = load_model(path + str(self) + f' {epoch}.keras')

    def save(self, path):
        self.model.save(path + str(self) + f' {self.epoch}.keras')
        with open(path + str(self) + f' {self.epoch} train.json', 'w') as history_file:
            json.dump(self.training_history[-1].history, history_file)

    def rate(self, doc):
        return self.model.predict(self.single(doc))[0][0] * 10

    def info(self):
        print(self.model.summary())

    def get_max_feature(self):
        return 2000 if self.max_feature == '2k' else NUM_VOCAB_NEW


class base(model):
    """docstring for base"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(base, self).__init__(category, max_feature, input_len, dataset)

    def batch(self, docs, golden_ratings, pre_done=None):
        MAX_FEATURE = self.get_max_feature()

        path = f'bins/processed review {self.dataset} {self.category}.md' if pre_done else None

        lemmatized_docs = self.pre_processor.docs_lemmatize(docs, path)

        lemmatized_docs_train, _, _, _, _, _ = self.pre_processor.split(lemmatized_docs, golden_ratings,
                                                                        self.TRAIN_SIZE, self.TEST_SIZE)

        word_index = self.pre_processor.make_index(lemmatized_docs_train)
        one_hot_docs = [[self.pre_processor.one_hot(word, MAX_FEATURE, word_index) for word in doc]
                        for doc in lemmatized_docs]

        self.X = sequence.pad_sequences(np.array(one_hot_docs, dtype=object), maxlen=self.get_input_len())
        self.y = np.array(golden_ratings)

    def single(self, doc):
        one_hot_doc = [self.pre_processor.one_hot(word, self.get_max_feature())
                       for word in self.pre_processor.lemmatize(doc)]
        return sequence.pad_sequences(np.array([one_hot_doc]), maxlen=self.get_input_len())

    def build(self):
        EMBEDDING_SIZE = 128
        HIDDEN_LAYER_SIZE = 64
        NUM_VOCAB = self.get_max_feature() + 2

        self.model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=self.get_input_len()))
        self.model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="mean_squared_error", optimizer="adam")

    def get_input_len(self):
        if self.input_len == 'avg':
            return INPUT_LENGTH_AVG_NEW
        else:
            return INPUT_LENGTH_MAX_OLD if self.dataset == 'old' else INPUT_LENGTH_MAX_NEW


class double_LSTM(model):
    """docstring for double_LSTM"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(double_LSTM, self).__init__(category, max_feature, input_len, dataset)

    def batch(self, docs, golden_ratings, pre_done=True):
        MAX_FEATURE = self.get_max_feature()

        # load processed data
        path = f'bins/processed review {self.dataset} {self.category}.md' if pre_done else None

        with open(path, encoding='utf-8') as md:
            lines = md.readlines()

        # data into 3D list (num_doc, num_sentence, num_word)
        docs = []
        doc = []
        for i in range(1, len(lines)):
            if lines[i][:8] == '# review':
                docs.append([line.split() for line in doc])
                doc = []
            else:
                doc.append(lines[i])
        docs.append([line.split() for line in doc])

        # one-hot encoding of word
        docs_train, _, _, _, _, _ = self.pre_processor.split(docs, golden_ratings, self.TRAIN_SIZE, self.TEST_SIZE)

        lemmatized_docs_train = [[word for sentence in doc for word in sentence] for doc in docs_train]

        word_index = self.pre_processor.make_index(lemmatized_docs_train)
        one_hot_docs = [[[self.pre_processor.one_hot(word, MAX_FEATURE, word_index) for word in sentence]
                         for sentence in doc] for doc in docs]

        # sentence padding
        for i in range(len(one_hot_docs)):
            num_pad = self.get_review_len() - len(one_hot_docs[i])
            one_hot_docs[i] = one_hot_docs[i][-num_pad:] if num_pad < 0 else [[]] * num_pad + one_hot_docs[i]
        # word padding
        one_hot_docs = [sequence.pad_sequences(np.array(doc, dtype=object), maxlen=self.get_sentence_len())
                        for doc in one_hot_docs]

        self.X = np.array([[word for sentence in doc for word in sentence] for doc in one_hot_docs])
        self.y = np.array(golden_ratings)

    def single(self, doc):
        one_hot_doc = [[self.pre_processor.one_hot(word, self.get_max_feature())
                        for word in self.pre_processor.sentence_lemmatize(sentence)]
                       for sentence in self.pre_processor.sent_tokenize(doc)]
        num_pad = self.get_review_len() - len(one_hot_doc)
        one_hot_doc = one_hot_doc[-num_pad:] if num_pad < 0 else [[]] * num_pad + one_hot_doc
        one_hot_doc = sequence.pad_sequences(np.array(one_hot_doc, dtype=object), maxlen=self.get_sentence_len())

        return np.array([[word for sentence in one_hot_doc for word in sentence]])

    def build(self):
        EMBEDDING_SIZE = 128
        SENTENCE_FEATURE = 32
        DOC_FEATURE = 16
        NUM_VOCAB = self.get_max_feature() + 2
        INPUT_LENGTH = self.get_review_len() * self.get_sentence_len()

        embedding_model = Sequential(name='embedding_model')
        embedding_model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=INPUT_LENGTH))

        sentence_model = Sequential(name='sentence_model')
        sentence_model.add(LSTM(SENTENCE_FEATURE, dropout=0.2, recurrent_dropout=0.2))

        doc_model = Sequential(name='doc_model')
        doc_model.add(LSTM(DOC_FEATURE, dropout=0.2, recurrent_dropout=0.2))
        doc_model.add(Dense(1, activation='sigmoid'))

        sentences = tf.reshape(embedding_model.output, shape=(-1, self.get_sentence_len(), EMBEDDING_SIZE))
        sentences = sentence_model(sentences)
        docs = tf.reshape(sentences, shape=(-1, self.get_review_len(), SENTENCE_FEATURE))
        ratings = doc_model(docs)

        self.model = Model(embedding_model.input, ratings, name=str(self))
        self.model.compile(loss="mean_squared_error", optimizer="adam")

    def get_sentence_len(self):
        if self.input_len == 'avg':
            return SENTENCE_LENGTH_AVG_OLD if self.dataset == 'old' else SENTENCE_LENGTH_AVG_NEW
        else:
            return SENTENCE_LENGTH_MAX_OLD if self.dataset == 'old' else SENTENCE_LENGTH_MAX_NEW

    def get_review_len(self):
        if self.input_len == 'avg':
            return REVIEW_LENGTH_AVG_OLD if self.dataset == 'old' else REVIEW_LENGTH_AVG_NEW
        else:
            return REVIEW_LENGTH_MAX_OLD if self.dataset == 'old' else REVIEW_LENGTH_MAX_NEW
