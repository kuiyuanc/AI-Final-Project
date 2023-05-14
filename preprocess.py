import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import os
import csv


def get_wordnet_pos(treebank_tag):
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


def lemmatize_sent(sent):
    result = []
    lemmatizer = WordNetLemmatizer()
    for w, pos in pos_tag(word_tokenize(sent)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        result.append(lemmatizer.lemmatize(w, pos=wordnet_pos))
    return result


"""
如果要執行程式必須將要import的csv檔存在data這個資料夾下
或更改下面'./data' and './data/'
"""

reviews = []
data_files = os.listdir('./data')
for file in data_files:
    csvfile = open('./data/' + file, newline='', encoding="utf-8")
    reader = csv.reader(csvfile)
    for name, text, rating in reader:
        if name == 'Anime':
            continue
        rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
        reviews.append([name, text, rate])
    csvfile.close()

preprocess_list = []
stop_words = set(stopwords.words("english"))
stop_words.update([',', '.', '\'\''])
for review in reviews:
    tempt = []
    sentences = sent_tokenize(reviews[0][1])
    for sentence in sentences:
        words = lemmatize_sent(sentence)
        for word in words:
            if word not in stop_words:
                tempt.append(word)
    preprocess_list.append(tempt)

freqs = []
for single_review in preprocess_list:
    freq = nltk.FreqDist(single_review)
    freqs.append(freq)
