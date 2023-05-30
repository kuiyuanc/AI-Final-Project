'''
    Run this file directly to give this project a quick try.
'''

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
    import csv

    # load reviews from reviews.csv
    # reviews: a list of list, each list in reviews contains 3 components
    #   - anime name
    #   - review body
    #   - review rating
    print('loading csv...')
    reviews = []
    # data_files = os.listdir('./data')
    # for file in data_files:
    # csvfile = open('./data/' + file, newline='', encoding="utf-8")
    dataset = 'new'
    with open(f'./data/review {dataset}.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for name, text, rating in reader:
            if name == 'Anime':
                continue
            rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
            reviews.append([name, text, rate])

    # turn reviews into valid input
    # names, text, ratings are corresponding columns of reviews
    print('transpose table...\n')
    names, texts, ratings = zip(*reviews)
    ratings = [rating / 10 for rating in ratings]

    r = rater()
    r.build(texts, ratings, None, input_len='max')
    r.load('base-all-max-new')

    # train & validate rater
    r.train('LSTM')

    # save rater
    # r.save()

    # print('\nrating of an positive AI article:')
    # positive_article = 'There are many positive aspects to AI! Here are just a few: 1. Improved efficiency: AI has the ability to process and analyze vast amounts of data quickly and accurately, leading to improved efficiency and productivity in various industries. 2. Personalization: AI algorithms can learn from user behavior and preferences, allowing for personalized recommendations and experiences. 3. Medical advances: AI has the potential to revolutionize healthcare by assisting doctors with diagnoses, predicting disease outbreaks, and developing new treatments. 4. Enhanced safety: AI can be used to monitor and detect potential safety hazards in real-time, such as detecting fraud or identifying potential security threats. 5. Accessibility: AI-powered technologies can provide accessibility features for people with disabilities, such as text-to-speech and speech recognition software. Overall, AI has the potential to positively impact many aspects of our lives, from healthcare and education to transportation and entertainment.'
    # scores = r.rate(positive_article)
    # for i in range(len(r.agents.keys())):
    #     print(f'\t{list(r.agents.keys())[i]}: {scores[i][0]}')

    # print('rating of a neutral AI article:')
    # neutral_article = 'While the development of AI brings numerous benefits and advancements, it also presents several drawbacks and challenges. Here are some common drawbacks associated with the development of AI: 1. Job Displacement: AI has the potential to automate tasks and replace human workers in various industries. This displacement of jobs can lead to unemployment and economic inequality if adequate measures are not taken to retrain and upskill the workforce. 2. Bias and Discrimination: AI systems are trained on existing data, which may contain biases and discrimination present in society. If not carefully addressed, these biases can be perpetuated and amplified by AI algorithms, leading to unfair or discriminatory outcomes, particularly in areas such as hiring, lending, and law enforcement. 3. Lack of Transparency: Some AI algorithms, such as deep learning neural networks, can be highly complex and difficult to interpret. This lack of transparency raises concerns about the accountability and ethical implications of AI decision-making, especially in critical domains like healthcare and autonomous vehicles. 4. Privacy Concerns: The increasing use of AI involves the collection and analysis of vast amounts of personal data. This raises concerns about privacy, as AI systems have the potential to intrude on individuals\' private lives and exploit sensitive information. Safeguarding privacy while leveraging the benefits of AI is a significant challenge. 5. Security Risks: AI systems can be vulnerable to cyber-attacks, manipulation, or misuse. Adversarial attacks, where malicious actors intentionally manipulate input data to deceive AI systems, pose a significant threat. Additionally, the reliance on AI for critical infrastructure and systems introduces new vulnerabilities that can be exploited by hackers. 6. Ethical Dilemmas: The development of AI raises complex ethical dilemmas. For example, questions arise regarding the moral responsibility and accountability for AI actions, the decision-making process during autonomous operations, and the potential development of lethal autonomous weapons. 7. Overreliance and Dependency: As AI becomes more capable and integrated into various aspects of society, there is a risk of overreliance and dependency on AI systems. If AI systems fail or make incorrect decisions, especially in critical scenarios like healthcare or transportation, the consequences could be severe. 8. Disruption of Social Dynamics: The widespread adoption of AI can disrupt social dynamics and interpersonal relationships. The integration of AI assistants and social robots, for instance, may impact human interactions, leading to potential isolation, detachment, and a decrease in empathy. Addressing these drawbacks requires careful consideration, proactive policies, and responsible development and deployment of AI systems to ensure that the benefits of AI are harnessed while minimizing its potential negative impacts.'
    # scores = r.rate(neutral_article)
    # for i in range(len(r.agents.keys())):
    #     print(f'\t{list(r.agents.keys())[i]}: {scores[i][0]}')

    article = 'Ask yourself this: what do you want from a sequel to The Legend of Zelda: Breath of the Wild? More enemy variety? Better dungeons? Totally unexpected new ideas? Or is simply more Hyrule to explore enough for you? Thankfully, you don’t have to pick just one, because Nintendo’s response to all of those answers is a casual but confident, “Sure thing.” The Legend of Zelda: Tears of the Kingdom doesn’t necessarily revolutionize what already made Breath of the Wild one of the greatest games of all time, but it’s not a sequel that’s simply more of the same, either. This sandbox is bigger, richer, and somehow even more ambitious, with creative new systems like vehicle building, ridiculous weapon crafting, and a revamped Hyrule map with a dizzying amount of depth further fleshing out the intoxicating exploration that made the original so captivating. Breath of the Wild felt far from unfinished but, inconceivably, Tears of the Kingdom has somehow made it feel like a first draft. Before we dive too deep into Hyrule, a quick note about spoilers. I won’t spoil the (actually pretty great) story Tears tells, but these games are about so much more than the plot. That magic the first time you see one of BotW’s dragons soaring overhead is around every corner here too, and the last thing I’d want to do is steal the many moments that made my jaw literally drop from you. That said, there are some huge parts of Tears that are introduced fairly early on that I will be talking about because of how fundamental they are to why this game is so impressive. I am going to preserve as much of the magic as I can but, if (like millions of others) you’ve already decided you are going to play Tears, you should probably just go play it and then come back to share in the wonder with me later.'
    score = r.rate(article, agent='LSTM')
    print(f'\nrating an review of Legend of Zelda: {score}')


def load_test():
    r = rater()
    r.load([['LSTM', 'models/']])

    article = 'Ask yourself this: what do you want from a sequel to The Legend of Zelda: Breath of the Wild? More enemy variety? Better dungeons? Totally unexpected new ideas? Or is simply more Hyrule to explore enough for you? Thankfully, you don’t have to pick just one, because Nintendo’s response to all of those answers is a casual but confident, “Sure thing.” The Legend of Zelda: Tears of the Kingdom doesn’t necessarily revolutionize what already made Breath of the Wild one of the greatest games of all time, but it’s not a sequel that’s simply more of the same, either. This sandbox is bigger, richer, and somehow even more ambitious, with creative new systems like vehicle building, ridiculous weapon crafting, and a revamped Hyrule map with a dizzying amount of depth further fleshing out the intoxicating exploration that made the original so captivating. Breath of the Wild felt far from unfinished but, inconceivably, Tears of the Kingdom has somehow made it feel like a first draft. Before we dive too deep into Hyrule, a quick note about spoilers. I won’t spoil the (actually pretty great) story Tears tells, but these games are about so much more than the plot. That magic the first time you see one of BotW’s dragons soaring overhead is around every corner here too, and the last thing I’d want to do is steal the many moments that made my jaw literally drop from you. That said, there are some huge parts of Tears that are introduced fairly early on that I will be talking about because of how fundamental they are to why this game is so impressive. I am going to preserve as much of the magic as I can but, if (like millions of others) you’ve already decided you are going to play Tears, you should probably just go play it and then come back to share in the wonder with me later.'
    score = r.rate(article, agent='LSTM')
    print(f'\nrating an review of Legend of Zelda: {score}')


if __name__ == '__main__':
    quick_try()
