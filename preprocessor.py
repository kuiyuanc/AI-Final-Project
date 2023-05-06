##################################
# corpora : body of text
# lexicon : words and their means
##################################
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

review = 'One of the many few anime and manga that I actually like and follow enthusiastically, \
the plot is amazing, it feels a mix of 2-3 genres and the art is complementing it all.\
This story is very unique in itself, although i dont like most of the reincarnation mangas,\
this one caught my eye because of the tragic story that continues after the reincarnation, \
a MC with the only goal of finding and killing his father,\
some key moments of the anime is really what makes it hype worth it.\
Overall the plot, art and as well as the characters are all amazing, totally recommended.'

stop_words = set(stopwords.words("english"))
stop_words.add(',')
stop_words.add('.')
words = word_tokenize(review)

li = [word for word in words if word not in stop_words]

# get frequency
freq = nltk.FreqDist(li)

# show the graph
freq.plot()
