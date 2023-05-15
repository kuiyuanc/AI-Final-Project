from pre_processor import pre_processor


class rater:
    """docstring for rater"""

    def __init__(self):
        self.agent = None
        self.pre_processor = None

    def train(self, reviews, golden_ratings):
        pass

    def test(self, reviews, golden_ratings):
        pass

    def load(self, model):
        pass

    def save(self, path):
        pass

    def rate(self, review):
        pass
