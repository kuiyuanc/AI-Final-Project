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
        TRAIN_SIZE = 0.8
        TEST_SIZE = 0.1
        MAX_FEATURE = self.get_max_feature()

        path = f'bins/processed review {self.dataset}.md' if pre_done else None

        lemmatized_docs = self.pre_processor.docs_lemmatize(docs, path)

        lemmatized_docs_train, _, _, _, _, _ = self.pre_processor.split(lemmatized_docs, golden_ratings,
                                                                        TRAIN_SIZE, TEST_SIZE)

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

        self.model = Sequential()
        self.model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=self.get_input_len()))
        self.model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="mean_squared_error", optimizer="adam")

    def get_input_len(self):
        if self.input_len == 'avg':
            return INPUT_LENGTH_AVG_OLD if self.dataset == 'old' else INPUT_LENGTH_AVG_NEW
        else:
            return INPUT_LENGTH_MAX_OLD if self.dataset == 'old' else INPUT_LENGTH_MAX_NEW

    def get_max_feature(self):
        if self.max_feature == '2k':
            return 2000
        else:
            return NUM_VOCAB_OLD if self.dataset == 'old' else NUM_VOCAB_NEW


class double_LSTM(model):
    """docstring for double_LSTM"""

    def __init__(self, category, max_feature, input_len, dataset):
        super(double_LSTM, self).__init__(category, max_feature, input_len, dataset)

    def build(self, docs, golden_ratings):
        raise NotImplementedError
