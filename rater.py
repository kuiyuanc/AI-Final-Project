import json
from pre_processor import pre_processor
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense


class rater:
    """docstring for rater"""

    def __init__(self):
        self.pre_processor = pre_processor()
        self.agents = {}
        # self.training_history = {}
        self.training_history = {'LSTM': []}

    def build(self, docs, golden_ratings, agent, max_feature=10**1000, input_len='avg'):
        print('pre-processor loading...')
        self.pre_processor.load(docs, golden_ratings)

        self.max_feature = min(max_feature, self.pre_processor.num_vocab())
        print(f'max feature = {self.max_feature}')

        if input_len == 'max':
            self.input_len = self.pre_processor.max_doc_len()
        else:
            self.input_len = self.pre_processor.avg_doc_len()
        print(f'input len = {self.input_len}')

        print('model building...\n')
        if agent == 'LSTM':
            EMBEDDING_SIZE = 128
            HIDDEN_LAYER_SIZE = 64
            NUM_VOCAB = self.max_feature + 2

            self.agents[agent] = Sequential()
            self.agents[agent].add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=self.input_len))
            self.agents[agent].add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
            self.agents[agent].add(Dense(1, activation='sigmoid'))
            self.agents[agent].compile(loss="mean_squared_error", optimizer="adam")

    def train(self, agent):
        TRAIN_SIZE = 0.8
        TEST_SIZE = 0.1
        BATCH_SIZE = 32
        NUM_EPOCHS = 10
        VERBOSE = 1

        for agent in self.agents:
            print('data formatting...\n')
            Xtrain, Xval, Xtest, ytrain, yval, ytest = self.pre_processor.batch(
                agent, self.max_feature, self.input_len, TRAIN_SIZE, TEST_SIZE)

            print('train:')
            # self.training_history[agent] = self.agents[agent].fit(
            #     Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xval, yval))
            for x in range(6, NUM_EPOCHS):
                print(f'epoch {x}:')
                history = self.agents[agent].fit(Xtrain, ytrain, batch_size=BATCH_SIZE, validation_data=(Xval, yval))
                self.training_history[agent].append(history)
                self.agents[agent].save(f'models/base-all-max-new/base-all-max-new {x}.keras')
                with open(f'models/base-all-max-new/base-all-max-new {x}.json', 'w') as f:
                    json.dump(self.training_history[agent][-1].history, f)
                exit()

            print('\ntest:')
            self.agents[agent].evaluate(Xtest, ytest, batch_size=BATCH_SIZE, verbose=VERBOSE)

    def load(self, name):
        self.agents['LSTM'] = load_model(f'models/{name}/{name} 5.keras')
        # for name in models:
        #     self.agents[name] = load_model(f'models/{name}.keras')
        #     self.training_history[name] = json.load(f'training history/{name}.json')
        # with open('bins/rater.txt', 'rb') as rater_info:
        #     info = json.load(rater_info)
        #     self.max_feature = info['max_feature']
        #     self.input_len = info['input_len']

    def save(self):
        for name, agent in self.agents.items():
            print(f'saving {name} to models/{name}...')
            agent.save(f'models/{name}.keras')
        for name, history in self.training_history.items():
            print(f'saving {name} training history to training history/{name}...')
            with open(f'training history/{name}.json', 'wb+') as history_file:
                json.dump(history.history, history_file)
        with open('bins/rater.json', 'wb+') as rater_info:
            json.dump({'max_feature': self.max_feature, 'input_len': self.input_len}, rater_info)

    def rate(self, doc, agent):
        data = self.pre_processor.single(doc, agent, self.max_feature, self.input_len)
        return self.agents[agent](data)[0][0] * 10

    def get_agents(self):
        return self.agents.keys()
