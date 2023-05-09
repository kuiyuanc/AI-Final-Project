# Intro. to AI-Final-Project
We aim to develop a model capable of predicting the rating of anime reviews, with potential extensions to manga reviews and Mandarin Chinese reviews. 

In addition, we hope to build a second model that can assign tags to anime based on reviews. 

Our ultimate objective is to create an app that offers anime recommendations to users, suggesting similar animes if given an anime name as input and recommending anime that fulfill specific tags if given tag inputs.

## Realm
1. NLP
2. ML

## Data Source
[Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)

## Method
1. Compute the TF-IDF vector whereby each element corresponds to the TF-IDF value of each word in a given review.
2. Initialize the parameters randomly, such that the resulting vector has the same dimensions as the TF-IDF vector.
3. `deduced review rating = dot product of the TF-IDF vector and the parameter vector`
4. Utilize gradient descent to optimize the parameters.

## Evaluation
`loss function = 2nd-norm of the guess vector, where each element represents the inferred rating of a given review`