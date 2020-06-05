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

Data was collected from [Spotify Track Dataset](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db) from Kaggle and from Genius Lyrics website using the lyricsgenius wrapper by johnwmillr (link) utlising Genius APIs and BeautifulSoup webscraping.
 
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

## Data Preparation/Cleaning/Wrangling

## Model Selection

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

* Model testing on the <ins>features and CountVectorized lyrics</ins> indicated Logistic Regression and Random Forest Models yielded high cross validation scores of 0.302 and 0.433 respectively.

* Model testing on the <ins>features and TfidfVectorized lyrics</ins> Logistic Regression and Random Forest Models yielded high cross validation scores of 0.399 and 0.398 respectively.


### Large Scale Model Testing

Following a thourough GridSearchCV Procedure with multiple classification models (see Table below), it was found that Random Forest was performing the best out of the 16 models.


## Model Optimisation & Evaluation

## Results

## Analysis

## Further Steps

## Conclusion
