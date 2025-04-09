"""
Module that defines all tasks required for the ETL pipeline in Airflow.
Tasks:
1. Scrape tweets using snscrape
2. Add sentiment labels using polarity scores
3. Preprocess tweets (tokenization, stopword removal, lemmatization, stemming)
"""

import os
import sys
import pandas as pd
import snscrape.modules.twitter as sntwitter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
from utils import helper

# Download required NLTK assets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

stopwords_ = stopwords.words("english")


def scrap_raw_tweets_from_web(**kwargs) -> None:
    """
    Scrape tweets using snscrape and save as a raw Parquet file to S3.

    Required kwargs:
    - s3_hook (S3Hook)
    - bucket_name (str)
    - search_query (str)
    - tweet_limit (int)
    - raw_file_name (str)
    """
    try:
        query = kwargs["search_query"]
        limit = kwargs["tweet_limit"]
        bucket = kwargs["bucket_name"]
        file_name = kwargs["raw_file_name"]
        s3_hook = kwargs["s3_hook"]

        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            tweets.append([tweet.date, tweet.id, tweet.lang, tweet.user.username, tweet.content])

        df = pd.DataFrame(tweets, columns=['datetime', 'id', 'lang', 'username', 'raw_tweets'])
        df.to_parquet(file_name, index=False, engine="pyarrow")
        s3_hook.load_file(filename=file_name, key=file_name, bucket_name=bucket)

    except Exception as e:
        raise Exception("[ETL][SCRAPE] Failed to scrape or save tweets.") from e


def add_sentiment_labels_to_tweets(**kwargs) -> None:
    """
    Add sentiment labels based on polarity scores and save labeled tweets to S3.

    Required kwargs:
    - s3_hook (S3Hook)
    - bucket_name (str)
    - temp_data_path (str)
    - raw_file_name (str)
    - labelled_file_name (str)
    """
    try:
        path = kwargs["temp_data_path"]
        raw_file = kwargs["raw_file_name"]
        output_file = kwargs["labelled_file_name"]
        s3_hook = kwargs["s3_hook"]
        bucket = kwargs["bucket_name"]

        df = pd.read_parquet(f"{path}/{raw_file}", engine="pyarrow")
        df = df[df["lang"] == "en"]

        df["cleaned_tweets"] = df["raw_tweets"].apply(helper.remove_noise)
        df["polarity"] = df["cleaned_tweets"].apply(helper.calculate_polarity)
        df["sentiment"] = df["polarity"].apply(helper.assign_sentiment_labels)

        df.to_parquet(output_file, index=True, engine="pyarrow")
        s3_hook.load_file(filename=output_file, key=output_file, bucket_name=bucket)

    except Exception as e:
        raise Exception("[ETL][LABEL] Failed to add sentiment labels.") from e


def preprocess_tweets(**kwargs) -> None:
    """
    Preprocess tweets: lowercase, tokenize, remove stopwords, lemmatize, stem.
    Saves the final processed dataset to S3.

    Required kwargs:
    - s3_hook (S3Hook)
    - bucket_name (str)
    - temp_data_path (str)
    - labelled_file_name (str)
    - preprocessed_file_name (str)
    """
    try:
        path = kwargs["temp_data_path"]
        input_file = kwargs["labelled_file_name"]
        output_file = kwargs["preprocessed_file_name"]
        s3_hook = kwargs["s3_hook"]
        bucket = kwargs["bucket_name"]

        df = pd.read_parquet(f"{path}/{input_file}", engine="pyarrow").iloc[:, 1:]
        df["cleaned_tweets"] = df["cleaned_tweets"].astype(str).str.lower()
        df["tokenized_tweets"] = df["cleaned_tweets"].apply(word_tokenize)

        df["tokenized_tweets"] = df["tokenized_tweets"].apply(lambda tokens: helper.remove_stopwords(tokens, stopwords_))
        df = helper.remove_less_frequent_words(df)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        df["lemmatized_tweets"] = df["tokenized_tweets"].apply(
            lambda tokens: " ".join([lemmatizer.lemmatize(word) for word in tokens])
        )

        # Stemming
        stemmer = PorterStemmer()
        df["processed_tweets"] = df["lemmatized_tweets"].apply(
            lambda text: " ".join([stemmer.stem(word) for word in text.split()])
        )

        # Reorder columns and map labels
        df = df.reindex(columns=[col for col in df.columns if col != "sentiment"] + ["sentiment"])
        df["labels"] = df["sentiment"].map({"neutral": 0, "negative": 1, "positive": 2})

        print(f"[ETL][PREPROCESS] Completed preprocessing: {df.shape}")
        df.to_parquet(output_file, index=False, engine="pyarrow")
        s3_hook.load_file(filename=output_file, key=output_file, bucket_name=bucket)

    except Exception as e:
        raise Exception("[ETL][PREPROCESS] Failed to preprocess tweets.") from e
