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
    import os
    import csv

    # load reviews from reviews.csv
    # reviews: a list of list, each list in reviews contains 3 components
    #   - anime name
    #   - review body
    #   - review rating
    reviews = []
    # data_files = os.listdir('./data')
    # for file in data_files:
    # csvfile = open('./data/' + file, newline='', encoding="utf-8")
    csvfile = open('./data/review.csv', encoding="utf-8")
    reader = csv.reader(csvfile)
    for name, text, rating in reader:
        if name == 'Anime':
            continue
        rate = int(rating.replace('Reviewer’s Rating:', '').replace(' ', '').replace('\n', ''))
        reviews.append([name, text, rate])
    csvfile.close()
    print('loaded successfully\n')

    # turn reviews into valid input
    # names, text, ratings are corresponding columns of reviews
    names, texts, ratings = zip(*reviews)
    ratings = [rating / 10 for rating in ratings]

    # split the training set & testing set
    #   - training set: 90%
    #   - testing set: 10%
    Xtrain, Xtest, ytrain, ytest = train_test_split(texts, ratings, train_size=.9, random_state=42)

    # train & validate rater
    r = rater()
    print('training & validating:')
    r.train(Xtrain, ytrain)

    # save rater
    r.save('models')
    print('\nsaved successfully')

    # test rater
    print('\ntesting:')
    r.test(Xtest, ytest)

    print('\nrating of an positive AI article:')
    positive_article = 'There are many positive aspects to AI! Here are just a few: 1. Improved efficiency: AI has the ability to process and analyze vast amounts of data quickly and accurately, leading to improved efficiency and productivity in various industries. 2. Personalization: AI algorithms can learn from user behavior and preferences, allowing for personalized recommendations and experiences. 3. Medical advances: AI has the potential to revolutionize healthcare by assisting doctors with diagnoses, predicting disease outbreaks, and developing new treatments. 4. Enhanced safety: AI can be used to monitor and detect potential safety hazards in real-time, such as detecting fraud or identifying potential security threats. 5. Accessibility: AI-powered technologies can provide accessibility features for people with disabilities, such as text-to-speech and speech recognition software. Overall, AI has the potential to positively impact many aspects of our lives, from healthcare and education to transportation and entertainment.'
    scores = r.rate(positive_article)
    for i in range(len(r.agents.keys())):
        print(f'\t{list(r.agents.keys())[i]}: {scores[i][0]}')

    print('rating of a neutral AI article:')
    neutral_article = 'While the development of AI brings numerous benefits and advancements, it also presents several drawbacks and challenges. Here are some common drawbacks associated with the development of AI: 1. Job Displacement: AI has the potential to automate tasks and replace human workers in various industries. This displacement of jobs can lead to unemployment and economic inequality if adequate measures are not taken to retrain and upskill the workforce. 2. Bias and Discrimination: AI systems are trained on existing data, which may contain biases and discrimination present in society. If not carefully addressed, these biases can be perpetuated and amplified by AI algorithms, leading to unfair or discriminatory outcomes, particularly in areas such as hiring, lending, and law enforcement. 3. Lack of Transparency: Some AI algorithms, such as deep learning neural networks, can be highly complex and difficult to interpret. This lack of transparency raises concerns about the accountability and ethical implications of AI decision-making, especially in critical domains like healthcare and autonomous vehicles. 4. Privacy Concerns: The increasing use of AI involves the collection and analysis of vast amounts of personal data. This raises concerns about privacy, as AI systems have the potential to intrude on individuals\' private lives and exploit sensitive information. Safeguarding privacy while leveraging the benefits of AI is a significant challenge. 5. Security Risks: AI systems can be vulnerable to cyber-attacks, manipulation, or misuse. Adversarial attacks, where malicious actors intentionally manipulate input data to deceive AI systems, pose a significant threat. Additionally, the reliance on AI for critical infrastructure and systems introduces new vulnerabilities that can be exploited by hackers. 6. Ethical Dilemmas: The development of AI raises complex ethical dilemmas. For example, questions arise regarding the moral responsibility and accountability for AI actions, the decision-making process during autonomous operations, and the potential development of lethal autonomous weapons. 7. Overreliance and Dependency: As AI becomes more capable and integrated into various aspects of society, there is a risk of overreliance and dependency on AI systems. If AI systems fail or make incorrect decisions, especially in critical scenarios like healthcare or transportation, the consequences could be severe. 8. Disruption of Social Dynamics: The widespread adoption of AI can disrupt social dynamics and interpersonal relationships. The integration of AI assistants and social robots, for instance, may impact human interactions, leading to potential isolation, detachment, and a decrease in empathy. Addressing these drawbacks requires careful consideration, proactive policies, and responsible development and deployment of AI systems to ensure that the benefits of AI are harnessed while minimizing its potential negative impacts.'
    scores = r.rate(neutral_article)
    for i in range(len(r.agents.keys())):
        print(f'\t{list(r.agents.keys())[i]}: {scores[i][0]}')

    print('rating of an article showing the sorrow of seeing people dying in the war:')
    article = 'The Sorrow of Witnessing Lives Lost in War: A Tragic Human Reality Introduction: War, with its violent clashes and destructive nature, has always been a source of immense sorrow and grief. Throughout history, countless lives have been lost, families torn apart, and communities shattered by the ravages of armed conflicts. The sight of people dying in war evokes a profound sense of sorrow that strikes at the core of our humanity, reminding us of the immense tragedy that unfolds when diplomacy fails and violence prevails. A Heartbreaking Reality: The impact of war is far-reaching, leaving scars that extend beyond the battlefield. It is an agonizing spectacle to see lives extinguished in the prime of their existence, dreams unfulfilled, and futures obliterated. The sorrow of witnessing people dying in war is intensified by the understanding that behind each fallen soldier, there are grieving families, shattered friendships, and communities forever altered. Innocent civilians, caught in the crossfire, suffer the most in these conflicts. Men, women, and children are uprooted from their homes, their lives torn asunder, and their dreams shattered. The sight of their anguish, the wails of their grief, and the weight of their collective suffering paint a haunting picture of the true cost of war. Lost Potential and Unfulfilled Lives: What intensifies the sorrow even further is the contemplation of the untapped potential and talents that die with the victims of war. Among the fallen are brilliant minds, aspiring artists, dedicated caregivers, and future leaders whose contributions to society will forever remain unrealized. Their loss is not only a tragedy for their loved ones but also a loss for humanity as a whole. The Cycle of Sorrow: War breeds a vicious cycle of sorrow, perpetuating further violence and suffering. As lives are taken, grief and anger take root in the hearts of those left behind, fueling a desire for revenge or justice. The sorrow that accompanies war does not end with the death of individuals but echoes through generations, leaving a lasting impact on the collective memory and perpetuating a cycle of pain. A Call for Reflection and Change: In the face of such sorrow, it is imperative that we pause to reflect on the devastating consequences of war and seek ways to prevent it. Diplomacy, dialogue, and peaceful resolution should be prioritized to prevent the loss of precious lives and the infliction of sorrow upon countless families. Education, understanding, and empathy are key in breaking the cycle of violence and cultivating a culture of peace. Conclusion: The sorrow of witnessing people dying in war is a tragic reminder of the immense human cost that comes with armed conflicts. It highlights the urgent need for global unity and a commitment to finding peaceful solutions to conflicts. By acknowledging this sorrow and working towards a world where diplomacy and understanding prevail, we honor the memory of those lost and strive to ensure that future generations are spared from the heart-wrenching agony of war. Let us hold onto hope and tirelessly work towards a future where peace reigns supreme, sparing humanity from the sorrow of war\'s cruel embrace.'
    scores = r.rate(article)
    for i in range(len(r.agents.keys())):
        print(f'\t{list(r.agents.keys())[i]}: {scores[i][0]}')


if __name__ == '__main__':
    quick_try()
