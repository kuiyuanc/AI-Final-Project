import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import TextCollection
from tensorflow.keras.preprocessing import sequence


class pre_processor:
    """docstring for preProccesor"""

    def __init__(self, docs=None):
        if docs:
            self.load(docs)

        self.lemmatizer = WordNetLemmatizer()
        self.__STOP_WORDS = set(stopwords.words("english") + [',', '.', '\'\''])
        self.__PAD = "__PAD"
        self.__UNK = "__UNK"

    def load(self, docs):
        self.docs = docs

        self.lemmatized_docs = [self.lemmatize(doc) for doc in docs]

        self.bag_of_word = nltk.FreqDist([word for doc in self.lemmatized_docs for word in doc
                                          if word not in self.__STOP_WORDS])

        self.word_index = {x[0]: i + 2 for i, x in enumerate(self.bag_of_word.most_common())} | {
            self.__PAD: 0, self.__UNK: 1}

        self.one_hot_docs = [[self.one_hot(word) for word in doc] for doc in self.lemmatized_docs]

        self.text_collection = TextCollection(self.lemmatized_docs)

    def pre_process(self, doc):
        x = np.array([[self.one_hot(word) for word in self.lemmatize(doc)]])
        x = sequence.pad_sequences(x, maxlen=self.max_doc_len())  # pre-truncation/padding if too long/short
        return x

    def max_doc_len(self):
        return max(len(doc) for doc in self.lemmatized_docs)

    def num_vocab(self):
        return len(self.bag_of_word)

    def tf(self, word, doc_index):
        doc = ''.join([word + ' ' for word in self.lemmatized_docs[doc_index]])
        return self.text_collection.tf(word, doc)

    def idf(self, word):
        return self.text_collection.idf(word)

    def tf_idf(self, word, doc_index):
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

    def one_hot(self, word):
        return self.word_index[word] if word in self.word_index else self.word_index[self.__UNK]
