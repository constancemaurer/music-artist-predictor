# Music Artist Predictor

**General Assembly | Data Science Immersive | Capstone Project**

<img src="images/Screenshot 2020-06-05 at 11.12.01.png">

***

ML model that is able to predict the music artist of a track based on the song's Spotify music features and Genius lyrics.

## Table of Contents

* [Problem Statement](##Problem-Statement)
* [Goals](##Goals)
* [Methodology](##Methodolgy)
* [Python Libraries](##Python-Libraries)
* [Data Collection](##Data-Collection)
* [Data Preparation](##Data-Preparation)
* [Exploratory Data Analysis](##Exploratory-Data-Analysis)
* [Model Selection](##Model-Selection)
* [Model Optimisation](##Model-Optimisation)
* [Results](##Results)
* [Analysis](##Analysis)
* [Further Steps](##Further-Steps)
* [Conclusion](##Conclusion)

## Problem Statement

Can we predict a music artist by his/her songs' spotify music features, popularity score and lyrics?

## Goals

1. Create a multinominal classification model that is able to predict the music artist by the artists' songs given the songs' spotify audio features and lyrics of the song.
2. Evaluate the model by testing it on unseen data and producing classification report, confusion matrix, ROC curve and other evaluation metrics.

## Methodology

1. Select artists from a Spotify Dataset published on Kaggle.

2. Retrieve lyrics from artists' songs through Genius API wrapper called lyricsgenius by johnwmillr.

3. Establish SQL database.

4. Merge datasets.

5. Clean data of missing values, duplicates and other anomalies.

6. Perform sentimental analysis and other NLP analysis to derive NLP-based continous features.

7. Prepare training and testing set.

8. Prepare Count- and Tfidf-vectorised features.

9. Perform standarisation.

10. Select a model by testing different classifiers.

11. Optimise model.

12. Evaluate model with Accuracy, Precision and Recall.

12. Outline results, limitations and future steps.

## Python Libraries

[Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), [textacy](https://chartbeat-labs.github.io/textacy/build/html/index.html), [lexical diversity](https://pypi.org/project/lexical-diversity/), [VaderSentiment](https://github.com/Holek/vader_sentiment), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Postgres](https://postgresapp.com/), [Regex(re)](https://docs.python.org/3/library/re.html#module-re)

## Data Collection

Data was collected from [Spotify Track Dataset](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db) sourced from Kaggle and from Genius Lyrics website using the [lyricsgenius](https://github.com/johnwmillr/LyricsGenius) wrapper by johnwmillr utilising Genius APIs and BeautifulSoup webscraping.


Variable | Description | Data Type | Location
--- | --- | --- | ---
genre | The music genre of the song. | string object | {spotify_track,genius_lyrics}
artist_name | The artist or band name of the song. | string object | {spotify_track,genius_lyrics}
track_name | The track name is simply title of the song. | string object | {spotify_track}
track_id | The spotify ID number which at the time the dataset was collected functioned as the uri but might not match with current uris and IDs. | string object | {spotify_track}
popularity | Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note that the popularity value may lag actual popularity by a few days: the value is not updated in real time. | int64 | {spotify_track}
acousticness | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. | float64 | {spotify_track}
danceability | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. | float64 | {spotify_track}
duration_ms | The duration of the track in milliseconds. | int64 | {spotify_track}
energy | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. | float64 | {spotify_track}
instrumentalness | Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. | float64 | {spotify_track}
key | The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. | string object | {spotify_track}
liveness | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. | float64 | {spotify_track}
loudness | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. | float64 | {spotify_track}
mode | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0. | string object | {spotify_track}
speechiness | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. | float64 | {spotify_track}
tempo | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. | float64 | {spotify_track}
time_signature | An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). | string object | {spotify_track}
valence | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). | float 64 | {spotify_track}
release_year | The date the track/song was released. | string object - has to be changed to datetime | {genius_lyrics}
spotify_uri | Genius' record of spotify uri for a track. | string object | {genius_lyrics}
lyrics | Web scrapped lyrics from genius.com website. | string object | {genius_lyrics} 
 


## Data Preparation

**Data Cleaning & Processing**

* Removing duplicated entries

* Adjust artist name spellings of the Genius Data to the Spotify spelling

* Removing entries containing a featuring artist with the primary artist 

* Removing artist which aren't part of the target classes

* some fo the processing steps involved for feature engineering of the lextical diversity^, textacy stats^^ or sentiment analysis^^^ 
or CountVectorization(!) or TfidfVectorization(!!) of the Lyrics data:

     1. Removing elements such as [Chorus] or [Intro] added by the Lyrics Genius Website - utilising Regex. (all)

     2. Conversion to lowercase - utilising Python (!, !!)

     3. Parsing - performed by *textacy* or CountV/TfidifV (all)

     4. Tokenization - performed by *textacy* or skicit-learn's CountV and TfidfV classes (all)

     5. Lemmatization performed by *lexical diversity* and *Vader* (^,^^^)

***

**Features Engineered (excl. any CountVectorizer/TfidfVectorizer corpus)**

* Lexical diversity features Token-Type-Ratio (TTR) and Measure of Textual Lexical Diversity (MTLD) were engineered with [lexical diversity library](https://pypi.org/project/lexical-diversity/)

* NLP features number of sentences (n_sentences), word count (word_count), character count (character_count), number of syllables (n_syllables), unique word count (unique_word_count), number of long words (n_long_words), number of monosyllable words (n_monosyllable_words) and number of polysyllable words (n_polysyllable_words) were created using the [textacy library](https://chartbeat-labs.github.io/textacy/build/html/index.html)

* Sentiment Analysis and feature creation of vader_compound, vader_pos, vader_neu, vader_neg, objectivity_score and pos_vs_neg was performed using [VaderSentiment library](https://github.com/cjhutto/vaderSentiment).



<img src="images/Screenshot 2020-06-06 at 17.28.28.png" width=800>


## Exploratory Data Analysis

Please refer to the Exploratory Data Analysis Notebook and my Public Tableau Profile for further insights. This is a summarised EDA.

### Class imbalance

*bar graph depicting each artist and their class weight in percentage or decimal)*

### Correlation

*Correlation matrix plus small description*

### Sentiment Analysis

*top three negative artists*

*top three positive artists*

*vader compound vs valence*

*Adele vs Kendrick Lamar*



## Model Selection

**Baseline:** 0.058

* Modeling was performed on previously engineered features and lyrics, which were processed either through the use of scikit-learn's CountVectorizer (CountV), taking word counts, or TfidfVectorizer (TfidfV), where rare words are weighted more than common words.

* For both, CountV and TfidfV, a corpus of 1000 features and n grams with a range of 1 to 3 was created. Stop words were removed, but words were not stemmed and lemmatised, due to the use of slang in many tracks.

* Engineered features and CountV/TfidfV matrices were standarized using scikit-learn's StandardScaler() class.

* Modeling on PCA data was performed but had low scores and was therefore excluded from the testing process. 


### Preliminary Model Testing

First only four models (KNN, Logistic Regression, Random Forest, and Support Vector Machines) were tested with a GridsearchCV using only a few parameters per model on the engineered features and the lyrics processed with either scikit-learn's CountVectorizer or TfidfVectorizer.

**CountVectorizer Preliminary Model Test**
|Model               |Parameters           |Processing           |Train: Accuracy      |Train: Precision       |Train: Recall        |Train: F1            |Test: Accuracy        |Test: Precision      |Test: Recall         |Test: F1               |Cross-Val Score      |
|--------------------|---------------------|---------------------|---------------------|-----------------------|---------------------|---------------------|----------------------|---------------------|---------------------|-----------------------|---------------------|
|KNN                 |KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',                      metric_params=None, n_jobs=None, n_neighbors=5, p=2,                      weights='distance')|StandardScaler(), CountVectorizer()|1.000                |1.000                  |1.000                |1.000                |0.080                 |0.113                |0.074                |0.057                  |0.066                |
|Logistic Regression |LogisticRegression(C=7.896522868499725, class_weight=None, dual=False,                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,                    max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',                    random_state=None, solver='saga', tol=0.0001, verbose=0,                    warm_start=False)|StandardScaler(), CountVectorizer()|0.996                |0.998                  |0.994                |0.996                |0.304                 |0.239                |0.232                |0.223                  |0.302                |
|Random Forest       |RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',                        max_depth=40, max_features=50, max_leaf_nodes=None,                        min_impurity_decrease=0.0, min_impurity_split=None,                        min_samples_leaf=1, min_samples_split=2,                        min_weight_fraction_leaf=0.0, n_estimators=500,                        n_jobs=None, oob_score=False, random_state=42, verbose=0,                        warm_start=False)|StandardScaler(), CountVectorizer()|1.000                |1.000                  |1.000                |1.000                |0.452                 |0.445                |0.360                |0.361                  |0.433                |
|Support Vector Machine|SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',     max_iter=-1, probability=False, random_state=None, shrinking=True,     tol=0.001, verbose=False)|StandardScaler(), CountVectorizer()|0.994                |0.998                  |0.991                |0.994                |0.106                 |0.009                |0.055                |0.015                  |0.090                |

**TfidfVectorizer Preliminary Model Test**
|Model               |Parameters           |Processing           |Train: Accuracy      |Train: Precision       |Train: Recall        |Train: F1            |Test: Accuracy        |Test: Precision      |Test: Recall         |Test: F1               |Cross-Val Score      |
|--------------------|---------------------|---------------------|---------------------|-----------------------|---------------------|---------------------|----------------------|---------------------|---------------------|-----------------------|---------------------|
|KNN                 |KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',                      metric_params=None, n_jobs=None, n_neighbors=25, p=2,                      weights='distance')|StandardScaler(), TfidfVectorizer()|1.000                |1.000                  |1.000                |1.000                |0.103                 |0.187                |0.077                |0.067                  |0.104                |
|Logistic Regression |LogisticRegression(C=5.541020330009492, class_weight=None, dual=False,                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,                    max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',                    random_state=None, solver='saga', tol=0.0001, verbose=0,                    warm_start=False)|StandardScaler(), TfidfVectorizer()|1.000                |1.000                  |1.000                |1.000                |0.430                 |0.357                |0.364                |0.343                  |0.399                |
|Random Forest       |RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',                        max_depth=50, max_features=10, max_leaf_nodes=None,                        min_impurity_decrease=0.0, min_impurity_split=None,                        min_samples_leaf=1, min_samples_split=2,                        min_weight_fraction_leaf=0.0, n_estimators=500,                        n_jobs=None, oob_score=False, random_state=42, verbose=0,                        warm_start=False)|StandardScaler(), TfidfVectorizer()|1.000                |1.000                  |1.000                |1.000                |0.403                 |0.350                |0.290                |0.269                  |0.398                |
|Support Vector Machine|SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',     max_iter=-1, probability=False, random_state=None, shrinking=True,     tol=0.001, verbose=False)|StandardScaler(), TfidfVectorizer()|0.997                |0.999                  |0.994                |0.996                |0.061                 |0.027                |0.028                |0.008                  |0.063                |

**Preliminary findings**

* Model testing on the features and <ins>CountVectorized lyrics</ins> indicated Logistic Regression and Random Forest models yielded high cross validation scores of 0.302 and 0.433, respectively. 

* Model testing on the features and <ins>TfidfVectorized lyrics</ins> indicated Logistic Regression and Random Forest models yielded high cross validation scores of 0.399 and 0.398, respectively.



### Large Scale Model Testing

The large-scale model training was performed with TfidfV-engineered lyrics and the other features (see [Data Preparation](##Data-Preparation)).

Following a thourough GridSearchCV Procedure with multiple classification models (see Table below), it was found that Random Forest was performing the best out of the <ins>16 models</ins> tested.

|Model                                 |Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |Processing                         |Train: Accuracy|Train: Precision|Train: Recall|Train: F1|Test: Accuracy|Test: Precision|Test: Recall|Test: F1|Cross-Val Score|
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|---------------|----------------|-------------|---------|--------------|---------------|------------|--------|---------------|
|**Random Forest**                         |RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',                        max_depth=40, max_features=200, max_leaf_nodes=None,                        min_impurity_decrease=0.0, min_impurity_split=None,                        min_samples_leaf=1, min_samples_split=2,                        min_weight_fraction_leaf=0.0, n_estimators=500,                        n_jobs=None, oob_score=False, random_state=42, verbose=0,                        warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.437         |0.366          |0.341       |0.330   |**0.433**          |
|Bagging with Decision Tree            |BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None,                                                         criterion='gini',                                                         max_depth=None,                                                         max_features=None,                                                         max_leaf_nodes=None,                                                         min_impurity_decrease=0.0,                                                         min_impurity_split=None,                                                         min_samples_leaf=1,                                                         min_samples_split=2,                                                         min_weight_fraction_leaf=0.0,                                                         presort=False,                                                         random_state=42,                                                         splitter='best'),                   bootstrap=True, bootstrap_features=False, max_features=784,                   max_samples=0.9, n_estimators=100, n_jobs=None,                   oob_score=False, random_state=None, verbose=0,                   warm_start=False)|StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.426         |0.386          |0.343       |0.338   |0.400          |
|Logistic Regression L2                |LogisticRegression(C=0.21209508879201905, class_weight=None, dual=False,                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,                    max_iter=1000, multi_class='multinomial', n_jobs=None,                    penalty='l2', random_state=None, solver='lbfgs', tol=0.0001,                    verbose=0, warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.426         |0.356          |0.336       |0.328   |0.391          |
|Logistic Regression, Saga Solver      |LogisticRegression(C=7.443803013251689, class_weight=None, dual=False,                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,                    max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2',                    random_state=None, solver='saga', tol=0.0001, verbose=0,                    warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.433         |0.402          |0.369       |0.363   |0.387          |
|Linear Support Vector Classifier      |LinearSVC(C=0.001, class_weight=None, dual=True, fit_intercept=True,           intercept_scaling=1, loss='hinge', max_iter=1000,           multi_class='crammer_singer', penalty='l2', random_state=None,           tol=0.0001, verbose=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |StandardScaler(), TfidfVectorizer()|0.996          |0.997           |0.993        |0.995    |0.388         |0.304          |0.319       |0.299   |0.384          |
|Linear Support Vector Classifier      |LinearSVC(C=0.001, class_weight=None, dual=True, fit_intercept=True,           intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',           penalty='l2', random_state=None, tol=0.0001, verbose=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |StandardScaler(), TfidfVectorizer()|0.986          |0.993           |0.976        |0.983    |0.350         |0.236          |0.265       |0.235   |0.353          |
|Bagging with Best Decision Tree       |BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None,                                                         criterion='gini',                                                         max_depth=17,                                                         max_features=500,                                                         max_leaf_nodes=None,                                                         min_impurity_decrease=0.0,                                                         min_impurity_split=None,                                                         min_samples_leaf=1,                                                         min_samples_split=50,                                                         min_weight_fraction_leaf=0.0,                                                         presort=False,                                                         random_state=42,                                                         splitter='best'),                   bootstrap=True, bootstrap_features=False, max_features=887,                   max_samples=1.0, n_estimators=100, n_jobs=None,                   oob_score=False, random_state=None, verbose=0,                   warm_start=False)  |StandardScaler(), TfidfVectorizer()|0.700          |0.753           |0.592        |0.613    |0.369         |0.259          |0.264       |0.239   |0.338          |
|Support Vector Classifier - sigmoid   |SVC(C=0.27263157894736845, cache_size=200, class_weight=None, coef0=0.0,     decision_function_shape='ovr', degree=3, gamma=0.3333333333333333,     kernel='sigmoid', max_iter=-1, probability=False, random_state=None,     shrinking=True, tol=0.001, verbose=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |StandardScaler(), TfidfVectorizer()|0.140          |0.121           |0.107        |0.106    |0.278         |0.221          |0.218       |0.206   |0.302          |
|Logistic Regression L1                |LogisticRegression(C=2.442053094548651, class_weight=None, dual=False,                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,                    max_iter=1000, multi_class='ovr', n_jobs=None, penalty='l1',                    random_state=None, solver='liblinear', tol=0.0001, verbose=0,                    warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.338         |0.305          |0.295       |0.289   |0.293          |
|Decision Tree                         |DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=17,                        max_features=500, max_leaf_nodes=None,                        min_impurity_decrease=0.0, min_impurity_split=None,                        min_samples_leaf=1, min_samples_split=50,                        min_weight_fraction_leaf=0.0, presort=False,                        random_state=42, splitter='best')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |StandardScaler(), TfidfVectorizer()|0.414          |0.366           |0.335        |0.314    |0.247         |0.228          |0.211       |0.197   |0.241          |
|AdaBoost Classifier with Decision Tree|AdaBoostClassifier(algorithm='SAMME.R',                    base_estimator=DecisionTreeClassifier(class_weight=None,                                                          criterion='gini',                                                          max_depth=3,                                                          max_features=None,                                                          max_leaf_nodes=None,                                                          min_impurity_decrease=0.0,                                                          min_impurity_split=None,                                                          min_samples_leaf=1,                                                          min_samples_split=2,                                                          min_weight_fraction_leaf=0.0,                                                          presort=False,                                                          random_state=None,                                                          splitter='best'),                    learning_rate=0.1473684210526316, n_estimators=200,                    random_state=42)                                                                                      |StandardScaler(), TfidfVectorizer()|0.856          |0.929           |0.832        |0.863    |0.289         |0.301          |0.211       |0.211   |0.206          |
|Multi-layer Perceptron Classifier     |MLPClassifier(activation='identity', alpha=1e-10, batch_size=50, beta_1=0.9,               beta_2=0.999, early_stopping=True, epsilon=1e-08,               hidden_layer_sizes=(8, 8, 8), learning_rate='constant',               learning_rate_init=0.001, max_iter=1000, momentum=0.9,               n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,               random_state=42, shuffle=True, solver='adam', tol=0.0001,               validation_fraction=0.2, verbose=False, warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |StandardScaler(), TfidfVectorizer()|0.722          |0.694           |0.658        |0.650    |0.148         |0.101          |0.109       |0.101   |0.142          |
|KNN                                   |KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',                      metric_params=None, n_jobs=None, n_neighbors=25, p=2,                      weights='distance')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.103         |0.187          |0.077       |0.067   |0.104          |
|Bagging with Best KNN                 |BaggingClassifier(base_estimator=KNeighborsClassifier(algorithm='auto',                                                       leaf_size=30,                                                       metric='euclidean',                                                       metric_params=None,                                                       n_jobs=None,                                                       n_neighbors=25, p=2,                                                       weights='distance'),                   bootstrap=True, bootstrap_features=False, max_features=1025,                   max_samples=1.0, n_estimators=50, n_jobs=None,                   oob_score=False, random_state=None, verbose=0,                   warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.103         |0.179          |0.076       |0.061   |0.102          |
|Support Vector Classifier - poly      |SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',     max_iter=-1, probability=False, random_state=None, shrinking=True,     tol=0.001, verbose=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |StandardScaler(), TfidfVectorizer()|0.997          |0.999           |0.994        |0.996    |0.061         |0.027          |0.028       |0.008   |0.063          |
|Support Vector Classifier - rbf       |SVC(C=1.0615789473684212, cache_size=200, class_weight=None, coef0=0.0,     decision_function_shape='ovr', degree=3, gamma=0.1111111111111111,     kernel='rbf', max_iter=-1, probability=False, random_state=None,     shrinking=True, tol=0.001, verbose=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |StandardScaler(), TfidfVectorizer()|1.000          |1.000           |1.000        |1.000    |0.057         |0.001          |0.026       |0.003   |0.060          |
|Bagging with KNN                      |BaggingClassifier(base_estimator=KNeighborsClassifier(algorithm='auto',                                                       leaf_size=30,                                                       metric='minkowski',                                                       metric_params=None,                                                       n_jobs=None,                                                       n_neighbors=5, p=2,                                                       weights='uniform'),                   bootstrap=True, bootstrap_features=False, max_features=981,                   max_samples=0.8, n_estimators=50, n_jobs=None,                   oob_score=False, random_state=None, verbose=0,                   warm_start=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |StandardScaler(), TfidfVectorizer()|0.528          |0.927           |0.537        |0.618    |0.053         |0.084          |0.061       |0.035   |0.036          |

## Model Optimisation

blaaaaaa

## Results

**GridsearchCV: Random Forest**

* Achieved an optimal cross-validation score of X.XXX.

Feature importants based on coefficient absolute number

**Optimization of Random Forest Model:**

* ... increased the cross-valiation score to X.XXX.

* ... 

* ...

**Evaluation:**

* Accuracy score and classification report showed ...

* Precision score ...

* Recall score...

**Limitations:**

* 

* 

## Analysis

correclation between number of entries per artist and accuracy of the model

## Future Steps

## Conclusion
