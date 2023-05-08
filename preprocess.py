import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


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


review0 = """Animation - 8/10 Does very well in some parts but sadly, comes a bit short at others. The fights are consistently well animated for the most part. There's no end to the blood and gore when fighters are sliced and pierced, it's satisfying and tactful. The final episode was some of the best it had to offer and was a joy to behold. It seems it benefitted this studio dealing with more limiting sword-fighting and basic magic spells really allowed the studio to nail animation in crucial areas, ensuring every swing packs a punch, unmistakably they have learnt well from their previous work, "Darwin's Game". They also had prior work in "Rakudai Kishi" to hone their craft. I personally consider this(8/10) as the highest tier for less distinguished studios as it's Impossible for any non- "ufotable/David/Mappa/Madhouse/Wit/A1" show in 2023 to get past 8/10(in animation) as you risk insulting the truly peerless works from those studios."""

output = []
sentences = sent_tokenize(review0)
stop_words = set(stopwords.words("english"))
stop_words.update([',', '.','\'\''])

for sentence in sentences:
    words = lemmatize_sent(sentence)
    for word in words:
        if word not in stop_words:
            output.append(word)

# print(output)
freq = nltk.FreqDist(output)
freq.plot()
