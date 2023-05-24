from database import database
from crawler import crawler
from rater import rater
from ui import ui
from sklearn.model_selection import train_test_split


class anime_recommender:
    """docstring for animeRecommender"""

    def __init__(self):
        self.animes = None
        self.reviews = None
        self.models = None
        self.crawler = None
        self.rater = None
        self.ui = None

    def train(self):
        pass

    def build(self):
        pass

    def run(self):
        pass


def quick_try():
    import os
    import csv

    # load reviews from reviews.csv
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
    print('loaded successfully')

    # turn reviews into valid input
    names, texts, ratings = zip(*reviews)
    ratings = [rating / 10 for rating in ratings]
    Xtrain, Xtest, ytrain, ytest = train_test_split(texts, ratings, train_size=.9, random_state=42)

    # train & validate rater
    r = rater()
    print('validating:')
    r.train(Xtrain, ytrain)

    # test rater
    print('testing:')
    r.test(Xtest, ytest)

    print('rating of an positive AI article:')
    scores = r.rate('There are many positive aspects to AI! Here are just a few: 1. Improved efficiency: AI has the ability to process and analyze vast amounts of data quickly and accurately, leading to improved efficiency and productivity in various industries. 2. Personalization: AI algorithms can learn from user behavior and preferences, allowing for personalized recommendations and experiences. 3. Medical advances: AI has the potential to revolutionize healthcare by assisting doctors with diagnoses, predicting disease outbreaks, and developing new treatments. 4. Enhanced safety: AI can be used to monitor and detect potential safety hazards in real-time, such as detecting fraud or identifying potential security threats. 5. Accessibility: AI-powered technologies can provide accessibility features for people with disabilities, such as text-to-speech and speech recognition software. Overall, AI has the potential to positively impact many aspects of our lives, from healthcare and education to transportation and entertainment.')
    for i in range(len(r.agents.keys())):
        print(f'{r.agents.keys()[i]}: {scores[i]}')


if __name__ == '__main__':
    quick_try()
