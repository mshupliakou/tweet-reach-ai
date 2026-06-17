# Tweet Reach AI

A machine learning project for predicting social media engagement using NLP methods.

The goal of this project is to build a predictive model that estimates the expected interest in a tweet, measured by the number of retweets, based on the tweet text and selected metadata.

This project was prepared for the **Computational Intelligence Methods** course.

## Project Topic

**Prediction of interest in social media posts using NLP methods**

The project uses the **Trump Tweets** dataset from Kaggle:

https://www.kaggle.com/datasets/austinreese/trump-tweets

## Problem Description

The number of retweets is treated as a proxy for tweet popularity or audience interest. The task is formulated as a regression problem:

```text
Input: tweet text + metadata
Output: predicted number of retweets
```

Since the distribution of retweets is highly skewed and contains strong outliers, the model is trained on the transformed target:

```text
log1p(retweets)
```

The predictions can later be converted back to the original retweet scale using the inverse transformation.

## Project Scope

The project includes:

* loading and preparing the dataset,
* exploratory data analysis,
* text preprocessing,
* text representation using TF-IDF,
* feature engineering based on text, metadata and time-related variables,
* comparison of different dataset variants and feature sets,
* training regression models,
* model evaluation,
* model interpretability analysis using SHAP,
* discussion of results and limitations.

## Dataset

The dataset contains tweets posted by Donald Trump together with metadata such as publication date, tweet content, number of retweets, number of favorites, hashtags and mentions.

A major challenge in this dataset is **temporal drift**. Tweets from earlier years come from a period when the account had a much smaller reach. Therefore, the project compares two dataset variants:

* all available years,
* tweets from 2015 onward.

The best results were achieved using the `2015+` dataset variant.

## Project Pipeline

The general architecture of the solution is:

```text
Kaggle dataset
      |
      v
Data loading
      |
      v
Text cleaning and preprocessing
      |
      v
Chronological train / validation / test split
      |
      v
TF-IDF vectorization fitted only on the training set
      |
      v
Feature engineering
      |
      v
Model training
      |
      v
Evaluation
      |
      v
SHAP interpretability analysis
```

## Text Preprocessing

The tweet text was cleaned and normalized before vectorization. The preprocessing steps included:

* converting text to lowercase,
* removing URLs,
* removing user mentions,
* cleaning special characters,
* tokenization,
* removing stop words,
* lemmatization.

The processed text was then represented numerically using:

```text
TF-IDF Vectorizer
```

Both unigrams and bigrams were used.

## Features

Several feature sets were compared.

### Text only

Features based only on the TF-IDF representation of tweet content.

### Text + metadata without time

Text features combined with metadata such as:

* tweet length,
* number of words,
* presence of URLs,
* number of hashtags,
* number of mentions,
* presence of exclamation marks.

### Text + metadata + time

The most complete feature set, additionally including time-related features such as:

* `days_since_first`,
* `year`,
* day of week,
* weekend indicator.

This feature set achieved the best results.

## Models

The following regression models were tested:

* `DummyRegressor` as a simple baseline,
* `Ridge Regression`,
* `XGBoost Regressor`.

The final selected model configuration was:

```text
Dataset variant: 2015+
Feature set: text_meta_time
Model: XGBoost
```

## Results

The final model achieved the following results:

| Split      | RMSE log | MAE log | R2 log | MAE original scale | R2 original scale |
| ---------- | -------: | ------: | -----: | -----------------: | ----------------: |
| Validation |    0.408 |   0.304 |  0.109 |               5725 |             0.008 |
| Test       |    0.597 |   0.439 | -0.034 |               9449 |            -0.114 |

The best performance was obtained for the `2015+` dataset variant with the full feature set combining text, metadata and time-based features.

The results show that the model improves over a simple training-mean baseline. However, the R2 values close to zero or slightly below zero indicate that predicting the exact number of retweets is a difficult task. This is mainly caused by:

* viral outliers,
* temporal drift,
* changing account popularity over time,
* lack of information about the actual number of followers at the time of posting,
* unpredictable social and political context.

## SHAP Interpretability Analysis

The final XGBoost model was interpreted using **SHAP**.

The analysis showed that the most important feature was:

```text
days_since_first
```

This means that the model strongly relies on the time of publication. This is consistent with the observation that the account reach changed significantly over time, especially after 2015.

Important text-based features included words and phrases related to politics, media, public events and controversial topics.

The main interpretation is:

```text
Tweet content matters, but in this dataset the publication time and changing account reach have a very strong influence on the predicted number of retweets.
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/mshupliakou/tweet-reach-ai.git
cd tweet-reach-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies manually:

```bash
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib nltk kagglehub joblib jupyter
```

### 4. Run the notebook

```bash
jupyter notebook
```

Then open the main project notebook and run all cells from top to bottom.

## Technologies Used

* Python
* Jupyter Notebook
* pandas
* NumPy
* SciPy
* scikit-learn
* XGBoost
* SHAP
* Matplotlib
* NLTK
* KaggleHub

## Limitations

The main limitations of the project are:

* retweet count is difficult to predict due to viral outliers,
* the data is not stationary over time,
* the model does not have access to the number of followers at the time of posting,
* many text features are specific to one person and one historical period,
* the model may not generalize well to other accounts or future periods.

## Possible Improvements

Future improvements could include:

* using contextual embeddings such as BERT or Sentence Transformers,
* predicting engagement categories instead of exact retweet counts,
* adding historical follower count data,
* using more robust regression objectives,
* separate analysis of viral tweets,
* applying rolling time-based validation.

## Authors

Jak Kon, 
Michail Shupliakou, 
Krystsina Mironenka

Project prepared for the **Computational Intelligence Methods** course.

Repository:

https://github.com/mshupliakou/tweet-reach-ai
