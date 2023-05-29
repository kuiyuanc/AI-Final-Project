import json
import numpy as np
from pre_processor import pre_processor
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense


class model:
    """docstring for model"""

    def __init__(self, category, max_feature, input_len, dataset):
        self.category = category
        self.max_feature = max_feature
        self.input_len = input_len
        self.dataset = dataset

        self.pre_processor = pre_processor()
        self.model = Sequential()
        self.training_history = []

    def __str__(self):
        return self.category + '-' + str(self.max_feature) + '-' + self.input_len + '-' + self.dataset

    def batch(self, docs, golden_ratings):
        raise NotImplementedError

    def single(self, doc):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def train(self, path, start_epoch=0):
        TRAIN_SIZE = 0.8
        TEST_SIZE = 0.1
        BATCH_SIZE = 32
        NUM_EPOCHS = 10
        VERBOSE = 1

        print('data formatting...\n')
        X, y = self.batch()
        Xtrain, Xval, Xtest, ytrain, yval, ytest = self.pre_processor.split(X, y, TRAIN_SIZE, TEST_SIZE)

        print('train:')
        for i in range(start_epoch, NUM_EPOCHS):
            print(f'epoch {i}:')
            history = self.model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, validation_data=(Xval, yval))
            self.training_history.append(history)
            self.save(path + str(self) + f' {i}')

        print('\ntest:')
        self.model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE, verbose=VERBOSE)

    def load(self, name):
        print(f'{name} loading...')
        self.model = load_model(name + '.keras')
        # self.training_history = json.load(name + '.json')

    def save(self, name):
        print(f'{name} saving...')
        self.model.save(name + '.keras')
        with open(name + '.json', 'wb') as history_file:
            json.dump(self.training_history.history, history_file)
        with open(name + ' info.json', 'wb') as info_file:
            json.dump({'max_feature': self.max_feature, 'input_len': self.input_len}, info_file)

    def rate(self, doc):
        return self.model.predict(self.single(doc))[0][0] * 10


class LSTM(model):
    """docstring for LSTM"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(LSTM, self).__init__(category, max_feature, input_len, dataset)

    def batch(self, docs, golden_ratings):
        print('pre-processor loading...')
        self.pre_processor.load(docs)

        self.max_feature = min(self.max_feature, self.pre_processor.num_vocab())
        print(f'max feature = {self.max_feature}')

        self.input_len = self.pre_processor.max_doc_len() if self.input_len == 'max'\
            else self.pre_processor.avg_doc_len()
        print(f'input len = {self.input_len}')

        one_hot_docs = [[word if word < self.max_feature else self.pre_processor.__UNK for word in doc]
                        for doc in self.pre_processor.one_hot_docs]

        X = np.array(one_hot_docs, dtype=object)
        X = sequence.pad_sequences(X, maxlen=self.input_len)
        y = np.array(golden_ratings)
        return X, y

    def single(self, doc):
        one_hot_doc = [self.pre_processor.one_hot(word) if word < self.max_feature
                       else self.pre_processor.__UNK for word in self.pre_processor.lemmatize(doc)]
        return sequence.pad_sequences(np.array([one_hot_doc]), maxlen=self.input_len)

    def build(self):
        print('model building...\n')
        EMBEDDING_SIZE = 128
        HIDDEN_LAYER_SIZE = 64
        NUM_VOCAB = self.max_feature + 2

        self.model = Sequential()
        self.model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=self.input_len))
        self.model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="mean_squared_error", optimizer="adam")


class double_LSTM(model):
    """docstring for double_LSTM"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(double_LSTM, self).__init__(category, max_feature, input_len, dataset)

    def build(self, docs, golden_ratings):
        raise NotImplementedError
