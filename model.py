import json
import numpy as np
from pre_processor import pre_processor
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense


# before spliting
NUM_VOCAB_OLD = 18551
NUM_VOCAB_NEW = 108333
INPUT_LENGTH_AVG_OLD = 410
INPUT_LENGTH_AVG_NEW = 519
INPUT_LENGTH_MAX_OLD = 2565
INPUT_LENGTH_MAX_NEW = 7734


class model:
    """docstring for model"""

    def __init__(self, category, max_feature, input_len, dataset):
        self.category = category
        self.max_feature = max_feature
        self.input_len = input_len
        self.dataset = dataset
        self.epoch = 0

        self.pre_processor = pre_processor()
        self.model = Sequential()
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
        TRAIN_SIZE = 0.8
        TEST_SIZE = 0.1
        BATCH_SIZE = 32
        VERBOSE = 1

        Xtrain, Xval, Xtest, ytrain, yval, ytest = self.pre_processor.split(self.X, self.y,
                                                                            TRAIN_SIZE, TEST_SIZE)

        for i in range(self.epoch + 1, end_epoch + 1):
            print(f'epoch {i}:')
            history = self.model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, validation_data=(Xval, yval))
            self.training_history.append(history)
            self.epoch += 1
            self.save(path)

        print('testing...')
        self.model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE, verbose=VERBOSE)

    def load(self, path, epoch):
        self.epoch = epoch
        self.model = load_model(path + str(self) + f' {epoch}.keras')

    def save(self, path):
        self.model.save(path + str(self) + f' {self.epoch}.keras')
        with open(path + str(self) + f' {self.epoch}.json', 'w') as history_file:
            json.dump(self.training_history[-1].history, history_file)

    def rate(self, doc):
        return self.model.predict(self.single(doc))[0][0] * 10


class base(model):
    """docstring for base"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(base, self).__init__(category, max_feature, input_len, dataset)

    def batch(self, docs, golden_ratings, pre_done=None):
        path = f'bins/processed review {self.dataset}.md' if pre_done else None
        self.pre_processor.load(docs, path)

        if self.max_feature == '2k':
            MAX_FEATURE = 2000
        elif self.dataset == 'old':
            MAX_FEATURE = NUM_VOCAB_OLD
        else:
            MAX_FEATURE = NUM_VOCAB_NEW

        if self.input_len == 'avg':
            INPUT_LENGTH = INPUT_LENGTH_AVG_OLD if self.dataset == 'old' else INPUT_LENGTH_AVG_NEW
        else:
            INPUT_LENGTH = INPUT_LENGTH_MAX_OLD if self.dataset == 'old' else INPUT_LENGTH_MAX_NEW

        one_hot_docs = [[word if word < MAX_FEATURE else 1 for word in doc]
                        for doc in self.pre_processor.one_hot_docs]

        X = np.array(one_hot_docs, dtype=object)
        self.X = sequence.pad_sequences(X, maxlen=INPUT_LENGTH)
        self.y = np.array(golden_ratings)

    def single(self, doc):
        one_hot_doc = [self.pre_processor.one_hot(word) if word < self.max_feature
                       else self.pre_processor.UNK for word in self.pre_processor.lemmatize(doc)]
        return sequence.pad_sequences(np.array([one_hot_doc]), maxlen=self.input_len)

    def build(self):
        EMBEDDING_SIZE = 128
        HIDDEN_LAYER_SIZE = 64

        if self.max_feature == '2k':
            NUM_VOCAB = 2000
        elif self.dataset == 'old':
            NUM_VOCAB = NUM_VOCAB_OLD
        else:
            NUM_VOCAB = NUM_VOCAB_NEW

        if self.input_len == 'avg':
            INPUT_LENGTH = INPUT_LENGTH_AVG_OLD if self.dataset == 'old' else INPUT_LENGTH_AVG_NEW
        else:
            INPUT_LENGTH = INPUT_LENGTH_MAX_OLD if self.dataset == 'old' else INPUT_LENGTH_MAX_NEW

        self.model = Sequential()
        self.model.add(Embedding(NUM_VOCAB + 2, EMBEDDING_SIZE, input_length=INPUT_LENGTH))
        self.model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="mean_squared_error", optimizer="adam")


class double_LSTM(model):
    """docstring for double_LSTM"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(double_LSTM, self).__init__(category, max_feature, input_len, dataset)

    def build(self, docs, golden_ratings):
        raise NotImplementedError
