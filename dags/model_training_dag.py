#!/usr/bin/env python3

"""
Airflow DAG to train a BiLSTM model using GPU acceleration inside a Docker container.
This pipeline:
1. Pulls preprocessed data from Snowflake (loaded by ETL DAG)
2. Triggers model training in a separate user-built Docker container with TensorFlow GPU support.

The training container is not part of the Airflow stack and is designed for modular GPU utilization.
"""

import os
import sys
from datetime import datetime
import pandas as pd
from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

# Local config helper
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config

config = Config()

@dag(
    dag_id="model_training",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    description="Train a BiLSTM model using GPU-enabled Docker container",
    doc_md="""
### Model Training DAG
This DAG fetches data from Snowflake and trains a deep learning model inside a GPU-powered Docker container.
""",
)
def model_training_pipeline_dag():

    @task(task_id="load_data_from_snowflake")
    def pull_data_from_snowflake(query: str) -> pd.DataFrame:
        """
        Pulls preprocessed data from Snowflake for model training.

        Args:
            query (str): SQL query to execute in Snowflake

        Returns:
            pd.DataFrame: Query results as a DataFrame
        """
        try:
            hook = SnowflakeHook(
                snowflake_conn_id="snowflake_conn",
                account=os.getenv("ACCOUNT"),
                warehouse=os.getenv("WAREHOUSE"),
                database=os.getenv("DATABASE"),
                schema=os.getenv("SCHEMA"),
                role=os.getenv("ROLE"),
            )

            cursor = hook.cursor().execute(query)
            df = cursor.fetch_pandas_all()
            return df

        except Exception as e:
            raise RuntimeError("[TRAINING DAG] Failed to fetch data from Snowflake") from e

        finally:
            if "cursor" in locals():
                cursor.close()
            hook.close()

    # GPU-powered training container (Docker image must be prebuilt and available locally)
    train_model = DockerOperator(
        task_id="train_model",
        image="model_training_tf:latest",  # Should be locally built Docker image
        command="python3 model_training.py",
        docker_url="unix://var/run/docker.sock",
        auto_remove=True,
        api_version="auto",
        mount_tmp_dir=False,  # prevent unwanted volume mounts
        # Optional volume mounts for training data or logs (uncomment if needed)
        # volumes=["/host/path:/container/path"],
    )

    # Set execution order
    pull_data_from_snowflake(config["misc"]["query"]) >> train_model


model_train_dag = model_training_pipeline_dag()
