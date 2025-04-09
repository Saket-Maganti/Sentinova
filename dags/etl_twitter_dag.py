#!/usr/bin/env python3

"""
Airflow DAG for full Twitter ETL pipeline:
1. Scrapes tweets using snscrape
2. Adds sentiment labels
3. Cleans and preprocesses the data
4. Loads final output to Snowflake for downstream use
"""

import os
import json
import sys
from datetime import datetime
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from snowflake.connector.pandas_tools import write_pandas
from airflow.models.connection import Connection

# Local project imports
from task_definitions.etl_task_definitions import (
    scrap_raw_tweets_from_web,
    preprocess_tweets,
    add_sentiment_labels_to_tweets,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config, Connections, load_dataframe

config = Config()

@dag(
    dag_id="etl",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    description="Monthly Twitter ETL pipeline for sentiment analysis"
)
def twitter_data_pipeline_dag_etl():

    @task(task_id="configure_connections")
    def set_connections():
        """Create S3 and Snowflake connections using Airflow Connection objects."""
        try:
            # AWS
            aws_creds = {
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "region_name": os.getenv("REGION")
            }
            aws_conn = Connection(
                conn_id="s3_connection",
                conn_type="S3",
                extra=json.dumps(aws_creds)
            )
            Connections(aws_conn).create_connections()

            # Snowflake
            sf_conn = Connection(
                conn_id="snowflake_conn",
                conn_type="Snowflake",
                host=os.getenv("HOST"),
                login=os.getenv("LOGIN"),
                password=os.getenv("PASSWORD")
            )
            Connections(sf_conn).create_connections()

        except Exception as e:
            raise RuntimeError("[DAG][Connection] Failed to create Airflow connections.") from e

    s3_hook = S3Hook(aws_conn_id=config["aws"]["connection_id"])

    scrap_tweets = PythonOperator(
        task_id="scrape_tweets",
        python_callable=scrap_raw_tweets_from_web,
        op_kwargs={
            "s3_hook": s3_hook,
            "bucket_name": config["aws"]["s3_bucket_name"],
            "search_query": config["tweets-scraping"]["search_query"],
            "tweet_limit": config["tweets-scraping"]["tweet_limit"],
            "raw_file_name": config["files"]["raw_file_name"]
        }
    )

    @task(task_id="download_from_s3")
    def download_from_s3(temp_data_path: str, file_name: str):
        """Download intermediate file from S3."""
        downloaded = s3_hook.download_file(
            key=file_name,
            bucket_name=config["aws"]["s3_bucket_name"],
            local_path=temp_data_path
        )
        target_path = os.path.join(temp_data_path, file_name)
        if downloaded != target_path:
            os.rename(downloaded, target_path)

    label_tweets = PythonOperator(
        task_id="label_sentiment",
        python_callable=add_sentiment_labels_to_tweets,
        op_kwargs={
            "s3_hook": s3_hook,
            "bucket_name": config["aws"]["s3_bucket_name"],
            "temp_data_path": config["aws"]["temp_data_path"],
            "raw_file_name": config["files"]["raw_file_name"],
            "labelled_file_name": config["files"]["labelled_file_name"]
        }
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_tweets",
        python_callable=preprocess_tweets,
        op_kwargs={
            "s3_hook": s3_hook,
            "bucket_name": config["aws"]["s3_bucket_name"],
            "temp_data_path": config["aws"]["temp_data_path"],
            "labelled_file_name": config["files"]["labelled_file_name"],
            "preprocessed_file_name": config["files"]["preprocessed_file_name"]
        }
    )

    @task(task_id="load_to_snowflake")
    def load_to_snowflake(processed_file: str, table_name: str):
        """Load processed file to Snowflake table."""
        try:
            sf_hook = SnowflakeHook(
                snowflake_conn_id="snowflake_conn",
                account=os.getenv("ACCOUNT"),
                warehouse=os.getenv("WAREHOUSE"),
                database=os.getenv("DATABASE"),
                schema=os.getenv("SCHEMA"),
                role=os.getenv("ROLE")
            )
            df = load_dataframe(processed_file)
            write_pandas(conn=sf_hook, df=df, table_name=table_name, quote_identifiers=False)
            print(f"[INFO] Loaded {len(df)} records into Snowflake table '{table_name}'.")

        except Exception as e:
            raise RuntimeError(f"[DAG][Snowflake] Failed to load to Snowflake: {e}")

        finally:
            sf_hook.close()

    # --- Task Flow ---
    set_connections() >> scrap_tweets
    scrap_tweets >> download_from_s3(config["aws"]["temp_data_path"], config["files"]["raw_file_name"]) >> label_tweets
    label_tweets >> download_from_s3(config["aws"]["temp_data_path"], config["files"]["labelled_file_name"]) >> preprocess_data
    preprocess_data >> load_to_snowflake(config["files"]["preprocessed_file_name"], config["misc"]["table_name"])


etl_dag = twitter_data_pipeline_dag_etl()
