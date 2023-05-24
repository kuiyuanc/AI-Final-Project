from pre_processor import pre_processor
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense


class rater:
    """docstring for rater"""

    def __init__(self):
        self.pre_processor = pre_processor()
        self.input_len = None
        self.agents = {}

    def train(self, docs, golden_ratings):
        # docs pre-process
        self.pre_processor.load(docs)

        self.input_len = self.pre_processor.max_doc_len()

        # data pre-process
        TRAIN_SIZE = 8 / 9
        RANDOM_STATE = 42

        X = np.array(self.pre_processor.one_hot_docs, dtype=object)
        X = sequence.pad_sequences(X, maxlen=self.input_len)  # pre-padding if too short
        y = np.array(golden_ratings)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

        self.train_LSTM(Xtrain, Xtest, ytrain, ytest)

    def train_LSTM(self, Xtrain, Xtest, ytrain, ytest):
        # build model
        # MAX_FEATURES = 2000
        EMBEDDING_SIZE = 128
        HIDDEN_LAYER_SIZE = 64
        # vocab_size = min(MAX_FEATURES, self.pre_processor.num_vocab()) + 2
        vocab_size = self.pre_processor.num_vocab() + 2

        layers = [
            Embedding(vocab_size, EMBEDDING_SIZE, input_length=self.input_len),
            LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2),
            Dense(1),
            Activation("sigmoid")
        ]
        self.agents['LSTM'] = Sequential(layers)
        self.agents['LSTM'].compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

        # train model
        BATCH_SIZE = 32
        NUM_EPOCHS = 10
        VERBOSE = 'auto'

        self.agents['LSTM'].fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=VERBOSE,
                                validation_data=(Xtest, ytest))

    def test(self, docs, golden_ratings):
        self.pre_processor.load(docs)

        X = np.array(self.pre_processor.one_hot_docs)
        X = sequence.pad_sequences(X, maxlen=self.input_len)  # pre-padding if too short
        y = np.array(golden_ratings)

        BATCH_SIZE = 32
        NUM_EPOCHS = 10
        VERBOSE = 'auto'
        self.agents['LSTM'].evaluate(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=VERBOSE)

    def load(self, models):
        pass

    def save(self, path):
        pass

    def rate(self, doc):
        return [round(agent(self.pre_processor.pre_process(doc))[0] * 10) for agent in self.agents.values()]
