#!/usr/bin/env python3

"""
Promotes the best-performing ML model to production using behavioral testing and evaluation.

This script:
1. Loads latest & production model versions from the MLflow registry (hosted on EC2).
2. Benchmarks both using:
    - Minimum Functionality Test (negation test)
    - Invariance Test (typos + contraction expansion)
    - Model evaluation on full test set
3. Pushes the better-performing model to "Production" stage in MLflow and archives the other.

Author: Jithin Sasikumar | Modified by Vaishnavi Polampalli
"""

import os
import sys
import mlflow
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from keras.utils import to_categorical
from transformers import BertTokenizer

import behavioral_test

# Project imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config, load_dataframe
from utils.prepare_data import Dataset

from typing import Tuple

config = Config()


@dataclass
class Productionalize:
    tracking_uri: str
    test_data: str
    model_name: str
    batch_size: int
    sequence_length: int
    num_classes: int = 3
    client: mlflow.MlflowClient = None
    test_dataframe: pd.DataFrame = None
    latest_version: int = None
    filter_string: str = "name LIKE 'sentiment%'"

    def __post_init__(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = mlflow.MlflowClient()
            self.latest_version = self.client.get_latest_versions(name=self.model_name)[0].version
            self.test_dataframe = load_dataframe(self.test_data)
        except Exception as e:
            print(f"[ERROR] Failed to initialize MLflow client or load test data: {e}")
            sys.exit(1)

    def get_all_registered_models(self) -> None:
        """
        Print all registered models and versions under the defined filter string.
        """
        print("\n📦 Registered Models:")
        for model in self.client.search_registered_models(filter_string=self.filter_string):
            for v in model.latest_versions:
                print(f" - Name: {v.name}, Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")

    def load_models(self) -> Tuple[tf.function, tf.function]:
        """
        Load latest and production TensorFlow models from MLflow registry.

        Returns:
            Tuple: (latest_model, production_model)
        """
        latest = mlflow.tensorflow.load_model(f"models:/{self.model_name}/{self.latest_version}")
        prod = mlflow.tensorflow.load_model(f"models:/{self.model_name}/production")
        return latest, prod

    def transform_data(self, df: pd.DataFrame, col_name: str = "cleaned_tweets") -> tf.data.Dataset:
        """
        Convert DataFrame to tokenized TensorFlow dataset for inference.

        Args:
            df (pd.DataFrame): Input text and label data.
            col_name (str): Name of text column.

        Returns:
            tf.data.Dataset
        """
        y = to_categorical(df["labels"], self.num_classes)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = Dataset(
            tokenizer=tokenizer,
            dataframe=df,
            labels=y,
            batch_size=self.batch_size,
            max_length=self.sequence_length,
            col_name=col_name
        ).encode_bert_tokens_to_tf_dataset()
        return dataset

    def benchmark_models(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Evaluate latest and production models using MFT, INV, and test dataset.

        Returns:
            Tuple: (latest_model_accuracies, production_model_accuracies)
        """
        latest_model, production_model = self.load_models()

        # --- Minimum Functionality Test (Negation) ---
        mft_df = load_dataframe("./scripts/test_data/sample_test_data_for_mft.parquet")
        negated_df = behavioral_test.min_functionality_test(mft_df)
        mft_dataset = self.transform_data(negated_df, col_name="negated_text")

        acc_latest_mft = behavioral_test.run("MFT_latest", latest_model, mft_dataset, negated_df)
        acc_prod_mft = behavioral_test.run("MFT_production", production_model, mft_dataset, negated_df)

        # --- Invariance Test (Typos + Expansions) ---
        inv_df = self.test_dataframe.tail(100).copy()
        inv_df["cleaned_tweets"] = inv_df["cleaned_tweets"].apply(behavioral_test.invariance_test)
        inv_dataset = self.transform_data(inv_df)

        acc_latest_inv = behavioral_test.run("Invariance_latest", latest_model, inv_dataset, inv_df)
        acc_prod_inv = behavioral_test.run("Invariance_production", production_model, inv_dataset, inv_df)

        # --- Evaluation on full test set ---
        test_dataset = self.transform_data(self.test_dataframe)
        latest_eval_score = latest_model.evaluate(test_dataset)
        prod_eval_score = production_model.evaluate(test_dataset)

        # Final accuracy tuples
        latest_scores = (acc_latest_mft, acc_latest_inv, latest_eval_score[1])
        prod_scores = (acc_prod_mft, acc_prod_inv, prod_eval_score[1])

        return latest_scores, prod_scores

    def push_new_model_to_production(
        self,
        latest_scores: Tuple[float, float, float],
        prod_scores: Tuple[float, float, float]
    ) -> bool:
        """
        Promote the latest model to production if it outperforms current production model.

        Returns:
            bool: True if promoted, else False
        """
        print("\n🧪 Benchmark Comparison:")
        print(f"Latest Model:     MFT={latest_scores[0]:.4f}, INV={latest_scores[1]:.4f}, Eval={latest_scores[2]:.4f}")
        print(f"Production Model: MFT={prod_scores[0]:.4f}, INV={prod_scores[1]:.4f}, Eval={prod_scores[2]:.4f}")

        if latest_scores > prod_scores:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.latest_version,
                stage="Production"
            )
            print("✅ Latest model has been promoted to Production.")
            return True
        else:
            print("⚠️ Latest model did not outperform the production model across all benchmarks.")
            return False


def main() -> None:
    p = Productionalize(
        tracking_uri=config["model-tracking"]["mlflow_tracking_uri"],
        test_data=config["files"]["test_data"],
        model_name=config["model-registry"]["model_name"],
        batch_size=config["train-parameters"]["batch_size"],
        sequence_length=config["train-parameters"]["sequence_length"]
    )

    latest_acc, prod_acc = p.benchmark_models()
    success = p.push_new_model_to_production(latest_acc, prod_acc)

    if success:
        p.get_all_registered_models()


if __name__ == "__main__":
    main()
