#!/usr/bin/env python3

"""
Train a Bi-directional LSTM with BERT tokenization using GPU-accelerated TensorFlow.
Training occurs locally (e.g., inside a Docker container), while model tracking is
handled by an MLflow server hosted on AWS EC2.

Author: Jithin Sasikumar | Modified by Vaishnavi Polampalli
"""

import os
import sys
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras import losses, optimizers, metrics
from transformers import BertTokenizer

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
from utils.helper import Config, load_dataframe
from utils.prepare_data import Dataset
from utils.model import BiLSTM_Model
from utils.experiment_tracking import MLFlowTracker

config = Config()

@dataclass
class TrainParameters:
    batch_size: int
    num_classes: int
    embedding_dim: int
    sequence_length: int
    num_epochs: int
    learning_rate: float

@dataclass
class TrackingParameters:
    experiment_name: str
    mlflow_tracking_uri: str
    run_name: str
    experiment: bool

class ModelTrainer:
    def __init__(self, training_args: TrainParameters, tracking_args: TrackingParameters):
        self.training_args = training_args
        self.tracking_args = tracking_args

    def check_and_set_gpu(self) -> tf.config.LogicalDevice:
        """
        Configures GPU for training. Falls back to CPU if unavailable.
        """
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                tf.config.set_visible_devices(gpus[0], "GPU")
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.list_logical_devices("GPU")
                print(f"[INFO] GPU enabled: {logical_gpus}")
                return logical_gpus
            else:
                print("[WARN] No GPU found. Running on CPU.")
                return []
        except Exception as e:
            raise RuntimeError("Failed to configure GPU. Check Docker/NVIDIA setup.") from e

    def train(self) -> None:
        """
        Executes the full training pipeline and logs the run via MLflow.
        """
        self.check_and_set_gpu()

        # Track run in MLflow
        tracker = MLFlowTracker(
            experiment_name=self.tracking_args.experiment_name,
            tracking_uri=self.tracking_args.mlflow_tracking_uri,
            run_name=self.tracking_args.run_name,
            experiment=self.tracking_args.experiment
        )
        tracker.log()

        # Load and preprocess dataset
        print("[INFO] Loading data...")
        df = load_dataframe("./preprocessed_tweets.parquet")[['cleaned_tweets', 'labels']].copy()
        df = df.iloc[:35000]  # or load from config
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["labels"])
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        y_train = to_categorical(train_df["labels"], num_classes=self.training_args.num_classes)
        y_test = to_categorical(test_df["labels"], num_classes=self.training_args.num_classes)

        # Tokenize using BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_dataset = Dataset(tokenizer, train_df, y_train, self.training_args.batch_size,
                                self.training_args.sequence_length, train=True).encode_bert_tokens_to_tf_dataset()
        test_dataset = Dataset(tokenizer, test_df, y_test, self.training_args.batch_size,
                               self.training_args.sequence_length, train=True).encode_bert_tokens_to_tf_dataset()

        # Build model
        print("[INFO] Initializing BiLSTM model...")
        model: Sequential = BiLSTM_Model(
            vocab_size=tokenizer.vocab_size,
            num_classes=self.training_args.num_classes,
            embedding_dim=self.training_args.embedding_dim,
            input_length=self.training_args.sequence_length
        ).create_model()

        model.compile(
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=self.training_args.learning_rate, epsilon=1e-8),
            metrics=[metrics.CategoricalAccuracy(name="accuracy")]
        )

        print("[INFO] Training started...")
        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=self.training_args.num_epochs,
            batch_size=self.training_args.batch_size
        )

        # Optional: Save model locally or to MLflow
        # model.save("./model_output/bilstm_model.h5")

        tracker.end()
        print("[INFO] Training completed and logged to MLflow.")


def main() -> None:
    training_args = TrainParameters(
        batch_size=config["train-parameters"]["batch_size"],
        num_classes=config["train-parameters"]["num_classes"],
        embedding_dim=config["train-parameters"]["embedding_dim"],
        sequence_length=config["train-parameters"]["sequence_length"],
        num_epochs=config["train-parameters"]["num_epochs"],
        learning_rate=config["train-parameters"]["learning_rate"]
    )

    tracking_args = TrackingParameters(
        experiment_name=config["model-tracking"]["experiment_name"],
        mlflow_tracking_uri=config["model-tracking"]["mlflow_tracking_uri"],
        run_name=config["model-tracking"]["run_name"],
        experiment=config["model-tracking"]["experiment"]
    )

    trainer = ModelTrainer(training_args, tracking_args)
    trainer.train()


if __name__ == "__main__":
    main()
