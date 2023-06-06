# Overview
This repo is built for my final project of Intro. to AI.
With the code and resources in this repo, you can train either a single-layer LSTM model or a double-layer LSTM model to do sentiment analysis.

# Get Started

## Environment
Please clone all the folders and files in this repo.
Python version : 3.11.3
Library : Please refer to [requirements.txt](https://github.com/kuiyuanc/AI-Final-Project/blob/main/requirements.txt)

## Hyperparameter
Some hyperparameters can be set by the users when building a model.
If you want to build a model, please refer to [Train](#Train)
- `category` : This is the type of model.
    - `'base'` : Single-layer LSTM (Baseline model)
    - `'double_LSTM'` : Double-layer LSTM (Our main approach)
- `max_feature` : The number of the vocabularies the model will recognize.
    - `'2k'` : The most common 2000 vocabularies
    - `'max'` : All vocabularies
- `input_len` : The required length of the input. For double-layer LSTM model, there is no option to apply different padding or truncation strategies for sentences and reviews.
    - `'avg'` : The required length of the input will be the average length of the sentences/reviews in the training set.
    - `'max'` : The required length of the input will be the maximum length of the sentences/reviews in the training set.

Some hyperparameters are hard codes. You can find following code in [model.py](https://github.com/kuiyuanc/AI-Final-Project/blob/main/model.py)
- The structure of single-layer LSTM model (baseline)
    ```
        def build(self):
            EMBEDDING_SIZE = 128
            TEXT_FEATURE = 64
            NUM_VOCAB = self.get_max_feature() + 2

            self.model.add(Embedding(NUM_VOCAB, EMBEDDING_SIZE, input_length=self.get_input_len()))
            self.model.add(LSTM(TEXT_FEATURE, dropout=0.2, recurrent_dropout=0.2))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss="mean_squared_error", optimizer="adam")
    ```
- The structure of double-layer LSTM model (our main approach)
    ```
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
    ```

## Train
You can train a model by yourself with following code in [main.py](https://github.com/kuiyuanc/AI-Final-Project/blob/main/main.py)
The trained model and the training history will be save in the folder 'models/model name'.
```
category = 'double_LSTM'
max_feature = '2k'
input_len = 'max'
start_epoch = 0
end_epoch = 0

train(category, max_feature, input_len, start_epoch, end_epoch)
```
- `category`, `max_feature`, `input_len` : Please refer to [Hyperparameter](#Hyperparameter)
- `'start_epoch'` : A non-zero integer. The same model of epoch `'start_epoch'-1` will be loaded to start training at epoch `'start_epoch'`. If you want to train a model from scratch, please set this parameter to be 0.
- `'end_epoch'` : A non-zero integer. The model will be trained until this epoch, which is also included.
- `train(category, max_feature, input_len, start_epoch, end_epoch)` : Save the trained model and the training history in the folder 'models/model name'.

## Test
You can test a trained model with following code in [main.py](https://github.com/kuiyuanc/AI-Final-Project/blob/main/main.py)
The testing history will be saved in the folder 'models/model name'
```
category = 'double_LSTM'
max_feature = '2k'
input_len = 'max'
epoch = 0

name = category + '-' + max_feature + '-' + input_len
arr = load_arr(category, max_feature, input_len, epoch)

arr.test(name)
```
- `category`, `max_feature`, `input_len` : Please refer to [Hyperparameter](#Hyperparameter)
- `epoch` : A non-zero integer. The epoch of the model you want to test.
- `name` : The name of the model you want to test.
- `load_arr(category, max_feature, input_len, epoch)` : This function returns a `anime_review_rater` object loaded with the model you want to test.
- `arr.test(name)` : Test the model of the name with the testing set, and save the testing history in the folder 'models/model name'.


## Use
You can use a trained model the analyze the sentiment of a text with following code in [main.py](https://github.com/kuiyuanc/AI-Final-Project/blob/main/main.py)
```
category = 'double_LSTM'
max_feature = '2k'
input_len = 'max'
epoch = 0

name = category + '-' + max_feature + '-' + input_len
arr = load_arr(category, max_feature, input_len, epoch)

text = 'Any text you want to analyze.'
score = arr.rate(name, text)
print(f'score of "{text}": {score}')
```
- `category`, `max_feature`, `input_len` : Please refer to [Hyperparameter](#Hyperparameter)
- `epoch`, `name`, `load_arr(category, max_feature, input_len, epoch)` : Please refer to [Test](#Test)
- `arr.rate(name, text)` : Use the model of the name to predict the sentiment of the text. This function returns an float number in [0, 10].
    - 0 means the text is negative.
    - 10 means the text is positive.

## Show Structure
You can have a look at the structure of a trained with following code in [main.py](https://github.com/kuiyuanc/AI-Final-Project/blob/main/main.py)
```
category = 'double_LSTM'
max_feature = '2k'
input_len = 'max'

name = category + '-' + max_feature + '-' + input_len
arr = load_arr(category, max_feature, input_len, epoch)

arr.info(name)
```
- `category`, `max_feature`, `input_len` : Please refer to [Hyperparameter](#Hyperparameter)
- `name`, `load_arr(category, max_feature, input_len, epoch)` : Please refer to [Test](#Test)
- `arr.info(name)` : This function print the structure of the model you want to check in standard output.