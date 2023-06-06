import json
import numpy as np
import tensorflow as tf
from pre_processor import pre_processor, load_processed_reviews
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras import Model


NUM_VOCAB = 108181

INPUT_LENGTH_AVG = 249
# INPUT_LENGTH_MAX = 4070
INPUT_LENGTH_MAX = 1734

SENTENCE_LENGTH_AVG = 10
# SENTENCE_LENGTH_MAX = 3804
SENTENCE_LENGTH_MAX = 60
REVIEW_LENGTH_AVG = 24
# REVIEW_LENGTH_MAX = 372
REVIEW_LENGTH_MAX = 156


class model:
    """docstring for model"""

    def __init__(self, category, max_feature, input_len):
        self.TRAIN_SIZE = 0.8
        self.TEST_SIZE = 0.1
        self.BATCH_SIZE = 32
        self.VERBOSE = 1

        self.category = category
        self.max_feature = max_feature
        self.input_len = input_len
        self.epoch = -1

        self.pre_processor = pre_processor()
        self.model = Sequential(name=str(self))
        self.training_history = []

    def __str__(self):
        return self.category + '-' + self.max_feature + '-' + self.input_len

    def batch(self, texts, golden_ratings, pre_done=None):
        raise NotImplementedError

    def single(self, text):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def train(self, path, end_epoch):
        Xtrain, Xval, _, ytrain, yval, _ = self.pre_processor.split(self.X, self.y,
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

    def rate(self, text):
        return self.model.predict(self.single(text))[0][0] * 10

    def info(self):
        print(self.model.summary())

    def get_max_feature(self):
        return 2000 if self.max_feature == '2k' else NUM_VOCAB


class base(model):
    """docstring for base"""

    def __init__(self, category, max_feature, input_len):
        super(base, self).__init__(category, max_feature, input_len)

    def batch(self, texts, golden_ratings, pre_done=False):
        MAX_FEATURE = self.get_max_feature()

        lemmatized_texts = load_processed_reviews() if pre_done else self.pre_processor.texts_lemmatize(texts)
        lemmatized_texts = [[word for sentence in text for word in sentence] for text in texts]

        lemmatized_texts_train, _, _, _, _, _ = self.pre_processor.split(lemmatized_texts, golden_ratings,
                                                                         self.TRAIN_SIZE, self.TEST_SIZE)

        word_index = self.pre_processor.make_index(lemmatized_texts_train)
        one_hot_texts = [[self.pre_processor.one_hot(word, MAX_FEATURE, word_index) for word in text]
                         for text in lemmatized_texts]

        self.X = sequence.pad_sequences(np.array(one_hot_texts, dtype=object), maxlen=self.get_input_len())
        self.y = np.array(golden_ratings)

    def single(self, text):
        text = [word for sentence in self.pre_processor.text_lemmatize(text) for word in sentence]
        one_hot_text = [self.pre_processor.one_hot(word, self.get_max_feature()) for word in text]
        return sequence.pad_sequences(np.array([one_hot_text]), maxlen=self.get_input_len())

    def build(self):
        EMBEDDING_SIZE = 128
        TEXT_FEATURE = 64
        NUM_VOCAB = self.get_max_feature() + 2

        self.model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=self.get_input_len()))
        self.model.add(LSTM(TEXT_FEATURE, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="mean_squared_error", optimizer="adam")

    def get_input_len(self):
        return INPUT_LENGTH_AVG if self.input_len == 'avg' else INPUT_LENGTH_MAX


class double_LSTM(model):
    """docstring for double_LSTM"""

    def __init__(self, category, max_feature, input_len):
        super(double_LSTM, self).__init__(category, max_feature, input_len)

    def batch(self, texts, golden_ratings, pre_done=False):
        MAX_FEATURE = self.get_max_feature()

        texts = load_processed_reviews() if pre_done else self.pre_processor.texts_lemmatize(texts)

        texts_train, _, _, _, _, _ = self.pre_processor.split(texts, golden_ratings, self.TRAIN_SIZE, self.TEST_SIZE)

        word_index = self.pre_processor.make_index([[word for sentence in text for word in sentence]
                                                    for text in texts_train])
        one_hot_texts = [[[self.pre_processor.one_hot(word, MAX_FEATURE, word_index) for word in sentence]
                          for sentence in text] for text in texts]

        # sentence padding
        for i in range(len(one_hot_texts)):
            num_pad = self.get_review_len() - len(one_hot_texts[i])
            one_hot_texts[i] = one_hot_texts[i][-num_pad:] if num_pad < 0 else [[]] * num_pad + one_hot_texts[i]
        # word padding
        one_hot_texts = [sequence.pad_sequences(np.array(text, dtype=object), maxlen=self.get_sentence_len())
                         for text in one_hot_texts]

        self.X = np.array([[word for sentence in text for word in sentence] for text in one_hot_texts])
        self.y = np.array(golden_ratings)

    def single(self, text):
        one_hot_text = [[self.pre_processor.one_hot(word, self.get_max_feature())
                         for word in self.pre_processor.sentence_lemmatize(sentence)]
                        for sentence in self.pre_processor.sent_tokenize(text)]
        num_pad = self.get_review_len() - len(one_hot_text)
        one_hot_text = one_hot_text[-num_pad:] if num_pad < 0 else [[]] * num_pad + one_hot_text
        one_hot_text = sequence.pad_sequences(np.array(one_hot_text, dtype=object), maxlen=self.get_sentence_len())

        return np.array([[word for sentence in one_hot_text for word in sentence]])

    def build(self):
        EMBEDDING_SIZE = 128
        SENTENCE_FEATURE = 32
        TEXT_FEATURE = 16
        NUM_VOCAB = self.get_max_feature() + 2
        INPUT_LENGTH = self.get_review_len() * self.get_sentence_len()

        embedding_model = Sequential(name='embedding_model')
        embedding_model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=INPUT_LENGTH))

        sentence_model = Sequential(name='sentence_model')
        sentence_model.add(LSTM(SENTENCE_FEATURE, dropout=0.2, recurrent_dropout=0.2))

        text_model = Sequential(name='text_model')
        text_model.add(LSTM(TEXT_FEATURE, dropout=0.2, recurrent_dropout=0.2))
        text_model.add(Dense(1, activation='sigmoid'))

        sentences = tf.reshape(embedding_model.output, shape=(-1, self.get_sentence_len(), EMBEDDING_SIZE))
        sentences = sentence_model(sentences)
        texts = tf.reshape(sentences, shape=(-1, self.get_review_len(), SENTENCE_FEATURE))
        ratings = text_model(texts)

        self.model = Model(embedding_model.input, ratings, name=str(self))
        self.model.compile(loss="mean_squared_error", optimizer="adam")

    def get_sentence_len(self):
        return SENTENCE_LENGTH_AVG if self.input_len == 'avg' else SENTENCE_LENGTH_MAX

    def get_review_len(self):
        return REVIEW_LENGTH_AVG if self.input_len == 'avg' else REVIEW_LENGTH_MAX
