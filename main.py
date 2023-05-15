from database import database
from crawler import crawler
from rater import rater
from ui import ui

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


if __name__ == '__main__':
    ar = anime_recommender()
    ar.train()
    ar.run()
