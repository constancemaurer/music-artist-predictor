# music-artist-predictor

**General Assembly | Data Science Immersive | Capstone Project**

<img src="images/Screenshot 2020-06-05 at 11.12.01.png">


general discription

## Problem Statement

Can we predict a music artist by his/her songs' spotify music features, popularity score and lyrics?

## Goals

1.
2.


## Methodology

1. Select artists from a Spotify Dataset published on Kaggle.

2. Retrieve lyrics from artists' songs through Genius API and webscrapping with BeautifulSoup. Alternatively with a Python API wrapper called lyricsgenius by johnwmillr.

3. Establish SQL database

4. Merge datasets.

5. Clean data of missing values, duplicates and other anomalies.

6. Perform sentimental analysis and other NLP analysis to derive NLP-based continous features.

7. Prepare training and testing set.

8. Prepare count and TF-IDF vectorised features.

9. Perform standarisation.

10. *Perform Principal Component Analysis to establish feature importance and reduce the number of features.*

11. *STATSMODELS?*

12. *Cluster Analysis*

13. Model Selection by testing different Classifiers.

14. Model Optimisation and Evaluation with Accuracy, Precision and Recall.

15. Findings and Limitations

## Python Libraries


## Data Collection

Data was collected from Spotify Track Dataset from Kaggle (link) and from Genius Lyrics website using the lyricsgenius wrapper by johnwmillr
 (link) utlising Genius APIs and BeautifulSoup webscraping.
 
 feature table
 
 
## Data Preparation/Cleaning/Wrangling

## Model Selection

First only four models (KNN, Logistic Regression, Random Forest, and Support Vector Machines) were tested with a GridsearchCV using only a few parameters per model on the engineered features and the lyrics processed with either scikit-learn's CountVectorizer or TfidfVectorizer.

It was found that the ...Vectorizer seemed to yield better Cross Validation Scores.

Table of tfidf and cvec from first testing.


Following a thourough GridSearchCV Procedure with multiple classification models (see Table below), it was found that Random Forest was performing the best out of the 16 models.

Big model table---

## Model Optimisation & Evaluation

## Results

## Analysis

## Further Steps

## Conclusion
