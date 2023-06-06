# Introduction

With the rise of the Covid-19 pandemic, an increasing number of individuals are turning to anime as a form of entertainment from the comfort of their homes. Undoubtedly, watching anime serves as a source of relaxation. However, a disappointing anime can easily spoil one's mood entirely.

To address this concern, we are endeavoring to develop a model capable of generating ratings based on provided anime reviews. This model can be further utilized, such as in the creation of an application that recommends anime to individuals seeking works similar to their favorites, while also possessing a good rating.

We hope that this model will aid us in avoiding unsatisfactory works. (Note: The model's applicability extends beyond anime and can encompass other forms of media, such as games, TV shows, movies, etc., given the availability of sufficient data.)

# Related work

LSTM (Long Short-Term Memory) is a special type of recurrent neural network (RNN) . It is the architecture of the LSTM model, which consists three main parts: forget gate, input gate, and output gate.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/LSTM%20architecture.png)
This picture comes from the paper “A Review of Recurrent Neural Networks: LSTM Cells and Network Architectures.”

# Dataset/Platform

The data we used is got from the [Myanimelist](https://myanimelist.net/reviews.php?t=anime&filter_check=&filter_hide=&preliminary=on&spoiler=off&p=2) website. We crawled Top200 rating animes’ information and their reviews as our dataset. And the data form we crawled is like [anime name, reviews, rating]. Store these informations into a csv file and then do some data preprocessing. The total dataset size is about 87MB (about 34,000 reviews in total).

## Natural Language Pro-Process

The review strings can not be directed used as an input of mathematical models, so it is necessary to transform the raw data into numerical form.

### Word Lemmatization

We apply lemmatization to transform each word into its original form. This involves removing tenses, plurals, and other inflections to obtain the base or dictionary form of the word.

### One Hot Encoding

One-hot encoding is a representation technique used to convert categorical variables into a binary vector format. It is commonly used in machine learning when dealing with categorical data. The vector contains all zeros except for the index corresponding to the category, which is set to 1. This ensures that each category is represented independently.

We tweak the method above for efficiency. A vocabulary is mapped to an integer, instead of a vector with

Beside, different inflections of the same word can be transformed into the same integer with the help of word lemmatization.

# Baseline

## Introduction

### Word Embedding

Word embedding is a technique used to represent words as dense vector representations in a continuous vector space, where similar words are represented by vectors that are closer together. It captures semantic and syntactic relationships between words, which enables machine learning models to understand and process natural language.

### RNN & LSTM

LSTM is a kind of Recurrent Neural Network. RNN takes sequential input. The characteristics of RNN is the feedback connections that allow it to retain information from previous steps in the sequence.

LSTMs is proposed for a common problem in traditional RNNs called the vanishing gradient problem. In vanilla RNNs, the gradients that are backpropagated through time tend to become very small or vanish over long sequences, making it difficult for the network to learn long-term dependencies.

LSTMs overcome this problem by introducing a memory cell, which is a separate component responsible for storing and updating information over time. The cell has three main components: an input gate, a forget gate, and an output gate. These gates control the flow of information into and out of the cell, allowing it to selectively remember or forget information based on the context.

Therefore, LSTM is widely used in tasks such as sentiment analysis. Since our task primarily involves sentiment analysis, we have chosen LSTM as our baseline model.

## Data Pre-Process

The inputs of a neural network should have a consistent shape. Hence, truncation or padding is essential before feeding the reviews into the models.

The length of a review is calculated as the number of words, so each review should have the same number of words. Reviews shorter than the required input length are pre-padded with a reserved word ‘__PAD’; while reviews longer than the specified input length, they are pre-truncated to meet the requirement.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/pre-process%20baseline.png)


## Implementation

The structure of our baseline model is as shown in the graph below. The layers of the model are imported from Keras.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/structure%20baseline.png)

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/conceptual%20baseline.png)

The shape of input is (number_sample, text_length), where each row represents a review and each element represents a word.

The embedding layer converts the one-hot encoded words into word vectors with 128 elements.

The LSTM layer processes the word vectors in a review as sequential inputs. It generates a vector with 64 elements as the output, representing 64 features extracted from the review. This layer is designed to extract features at each input, enabling it to understand the review word-by-word with utilizing both short-term and long-term memory.

The dense layer is a fully connected layer, so its output is a linear combination of the output from the LSTM layer. We choose the 'sigmoid' activation function for this layer to constrain the output between 0 and 1.

We estimate the rating of a review by multiplying the output of the model by 10.

# Main Approach

## Introduction

Our main approach builds upon the baseline model and introduces some modifications to enhance its performance.

### Why do you come up with this idea ?

We believe that natural language should be understood in a larger structure, beyond individual words.  We aim to capture the contextual information and dependencies that exist at a higher level, thus our approach involves building a model that predicts the rating of a review by utilizing LSTM at the sentence level.

### What supports your idea ?

While searching for relevant works, we came across papers discussing convolutional LSTM only. Although our idea differs from convolutional LSTM, we believe that both approaches share the objective of capturing higher-level features of a review.

## Data Pre-Process

Similar to the process in baseline model, but here are two levels.

The first level involves padding or truncation of words within sentences.

The second level involves padding or truncation of sentences within reviews. That is, the length of a review is determined by the number of the sentences it contains, rather than the number of words.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/pre-process%20main.png)

## Algorithm

1. Perform word embedding for all the words in the reviews.
    
    In this step, the input shape is (number_review, sentence_length * review_length), which is different from the shape of data being pre-processed. As a result, reshaping the input samples is necessary. Each 2D array representing a review is turned into a 1D vector by concatenating the rows in the 2D array representation.
    
    After input transform, this step is the same as the first step in the baseline model.
    
2. Extract features from the sentences at the word level with first LSTM layer.
    
    In this step, the input shape is (number_sentence, sentence_length), which is different from the shape of the output from the embedding layer. As a result, reshaping the input samples is necessary. Each 1D vector representing a review is turned into a 2D array by spliting the elements in the 1D vector back into rows.
    
    This step is similar to the second step in the baseline model, where feature extraction is performed on each sentence. However, in this case, since the input represents a sentence, the output becomes a feature vector that represents the features of that particular sentence.
    
3. Capture features from the reviews at the sentence level with second LSTM layer.
    
    The input shape of this step is (number_review, review_length), which is different from the output shape of step 2. Hence, a reshape of the samples is required. Each group of 'review_length' elements in the output represents a review, as each review consists of 'review_length' sentences. Therefore, the sequential input of the second LSTM layer is obtained by concatenating the sentences.
    
    The LSTM layer in this step generates a review feature vector based on the sentence feature vectors. Because the relationship between sentence and sentence feature vector is similar to the one between word and embedding word vector, we can interpret the output of the LSTM layer in this step as the features of a review captured at the sentence level.
    
4. Compute the linear combination of the extracted features.
    
    The dense layer performs a linear combination on the review feature vector and output a scalar.
    
5. Apply the 'sigmoid' activation function to constrain the output between 0 and 1.
    
    This step is the same as the final step in the baseline model.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/data%20transformation.png)

## Implementation

The structure of our main approach model is as shown in the graph below.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/structure%20main.png)

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/conceptual%20main.png)

This table shows the mapping between the layer in the actual model and the conceptual model.

| Layer | What does the layer do? | The component in structure diagram |
| --- | --- | --- |
| embedding | step 1 | Embedding |
| tf.reshape | reshape between step 1 and step 2 | Reshape Sentence |
| sentence_model | step 2 | LSTM Sentence |
| tf.reshape_1 | reshape between step 2 and step 3 | Reshape Text |
| doc_model | step 3 ~ 5 | LSTM Text + Dense |

# Evaluation Metric

## Qualitative

We find some extreme reviews from [anikore.jp](http://anikore.jp/) and utilize the models to estimate the potential rating given by the reviewers for the anime. In this scenario, we make the assumption that positive reviews would likely result in a high predicted rating, such as 8 out of 10; while negative reviews would yield a low rating, such as 2 out of 10.

## Quantitative

We choose Mean Square Error (MSE) as our loss function for the models. We vary the parameters of the model and compare their performance.

The MSE is calculated on the output of the models. To obtain the prediction, we multiply the output of the models by 10. As a result, the actual MSE value should be one hundred times larger than the calculated result due to the scaling factor.

# Result & Analysis

## Result of Experiment & Discussion & Analysis

We have conduct 3 types of experiments, and one of them is evaluated with the qualitative evaluation metric, when the other two are evaluated by MSE.

### Qualitative Evaluation

| Review | Prediction of our main approach | Prediction of baseline |
| --- | --- | --- |
| positive | 9.72/10 | 8.82/10 |
| neutral | 6.58/10 | 6.01/10 |
| negative | 5.33/10 | 5.14/10 |
1. Both our main approach model and the baseline model performed poorly on negative reviews. After seeing this weird result, we found an imbalance in our dataset, with only 5.8702% of reviews having a rating lower than 5 (0~4).This indicates a disparity between the number of highly-rated reviews and low-rated reviews. Consequently, both models exhibit a tendency to provide higher estimations.
2. From the testing results, we can conclude that the double LSTM is better than base LSTM at higher rating reviews. And the base LSTM do better at low rating reviews.

### Changing the Length Required by the LSTM Layers

The model with average/max length refers to a model that takes input with a length equal to the average/maximum review length among the reviews in the training set.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/avg%20v.%20max.png)

1. Both the baseline model and our main approach model have nearly identical performance with average length inputs.
    
    We believe it can be attributed to the fact that approximately a half of the review is truncated during the pre-process stage. Thus, the models are trained on incomplete data, which hinders their ability to accurately extract features from the original content.
    
    In the case of our main approach model, not only are some sentences in a review truncated, but some words within sentences reserved are also cut off. Consequently, extracting features from these truncated sentences does not provide any additional performance gain.
    
2. Our main approach model with max length performs better than the two with average length.
    
    Our main approach model with max length inputs considers the entire review without truncation. By "reading" through the entire review, this model is expected to better understand the review and should therefore outperform the two models with average length inputs, as indeed observed.
    

### Changing the Number of Word Used for Embedding

During our research on implementing the baseline model, we discovered an approach to boost one hot encoding. This method involves mapping frequently occurring words to unique integers, while treating all other words as a reserved word '__UNK'. One hot encoding can introduce not negligible  pre-process overhead, especially as the training dataset grows. This overhead impacts not only the training process but also the model's usage.

To address this concern, we conducted an experiment using two models. The first model recognizes all the vocabularies present in the training set, while the second model only recognizes the 2000 most common words. The purpose of this experiment was to evaluate the impact of limiting the vocabulary size on the model's performance.

![](https://github.com/kuiyuanc/AI-Final-Project/blob/main/bins/pictures/2k%20v.%20all.png)

Based on the graph, it is evident that model performance can be significantly improved when it recognizes all vocabularies rather than just the most common 2000 words. This improvement can be attributed to several factors. 

1. By considering all vocabularies, the model can gather a broader range of information from the reviews. This increased information can enhance the probability of identifying more features that enable more accurate rating decisions. 
2. Relying solely on the most common 2000 words may overlook essential information. Reviewers typically do not repeatedly use the same sentiment words, which are crucial for rating determination. Consequently, these sentiment words necessary for accurate ratings may not be included among the most common 2000 words, resulting in poorer performance.

Although utilizing all vocabularies requires additional computational resources and time, it can reduce the loss and leads to improved performance.

### Phenomenon in Common

1. It is noteworthy that no model has achieved a validation loss below 0.01, which is regarded as a significant milestone. If the loss falls below 0.01, it provides confidence that the model can estimate with an average error of less than 1.
    
    Our interpretation of this observation is as follows: Reviews are highly subjective, so there can be instances where the content of a review does not closely align with the given rating. Thus, it becomes challenging for our model to achieve a low loss.
    
2. After the fifth epoch, overfitting was observed across all models. 
    
    In light of this observation, we believe that setting the training epoch to 5 strikes a balance between efficiency and performance, aiming for optimal results.
    

## Limitation

While training our main approach model with max length, we encountered a memory issue during the pre-processing stage. This problem arose due to extreme cases in the training set. Attempting to pad all sentences and reviews to match the length of the longest sentence and review proved to be excessively costly.

However, we aimed to improve the performance of our main approach model by allowing it to process the entire sentence and review. To address this, we analyzed the distribution of sentence and review lengths in the training set. We discovered that by truncating only 0.3% of sentences and reviews, we could set an input length using the formula: $max = avg + 6\sigma$, where $avg$ represents the average length of a sentence/review, and $\sigma$ denotes the standard deviation of the sentence/review length. This compromise enabled us to train our main approach model on our devices.

Nonetheless, this situation highlights a limitation of our main approach model: the significant padding overhead required to achieve optimal performance. The need for padding affects both the training process and the model's usage, introducing considerable overhead.

# Reference

- image used in slide: Yu, Yong, et al. “A Review of Recurrent Neural Networks: LSTM Cells and Network Architectures.” *Neural Computation*, vol. 31, no. 7, 2019, pp. 1235–1270
    
    [Review of Recurrent Neural Networks: LSTM Cells and Network Architectures](https://doi.org/10.1162/neco_a_01199)
    
- dataset
    
    [Anime Reviews - MyAnimeList.net](https://myanimelist.net/reviews.php?t=anime&filter_check=&filter_hide=&preliminary=on&spoiler=off&p=2)
    
- reviews used for qualitative evaluation (test1.txt, test2.txt, test3.txt)
    
    [おすすめアニメ動画を感想評価/人気でランキング【あにこれβ】](https://www.anikore.jp/)
    

# Contribution of Each Member

| Name | Student ID | Contribution |
| --- | --- | --- |
| 陳奎元 | 110550035 | 40% |
| 俞柏帆 | 110550087 | 30% |
| 林英碩 | 110550117 | 30% |