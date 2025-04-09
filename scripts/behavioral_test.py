"""
This module defines and performs behavioral testing for a sentiment analysis model,
following the approach outlined in the paper:
"Beyond Accuracy: Behavioral Testing of NLP models with CheckList" by Ribeiro et al.

Implemented Tests:
- Minimum Functionality Test (MFT): tests handling of negation
- Invariance Test (INV): checks prediction consistency under semantically preserved perturbations

Note:
This is **not** model evaluation; it's **behavioral testing**.

References:
[1] https://arxiv.org/abs/2005.04118
[2] https://github.com/marcotcr/checklist
"""

import os
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from checklist.perturb import Perturb
from sklearn.metrics import accuracy_score

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')


def min_functionality_test(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a Minimum Functionality Test (MFT) by adding negations to input samples.

    This test checks how well the model handles negated expressions in text,
    such as converting "I am happy" → "I am not happy".

    Args:
        dataframe (pd.DataFrame): Original test dataset containing 'sample_text' and 'labels'.

    Returns:
        pd.DataFrame: A new DataFrame with negated text and the same labels.
    """
    original_texts = dataframe["sample_text"].tolist()
    labels = dataframe["labels"].tolist()
    piped_texts = list(nlp.pipe(original_texts))

    perturbed = Perturb.perturb(piped_texts, Perturb.add_negation)
    negated_texts = [pair[1] for pair in perturbed.data]

    return pd.DataFrame({
        "negated_text": negated_texts,
        "labels": labels
    })


def invariance_test(text: str) -> str:
    """
    Applies invariance-preserving perturbations to test model robustness.

    The goal is for the model to maintain the same prediction even if the text
    contains small changes like typos or expanded contractions.

    Args:
        text (str): Original input text.

    Returns:
        str: Perturbed version of the input text.
    """
    text_with_typo = str(Perturb.add_typos(text))
    return str(Perturb.expand_contractions(text_with_typo))


def run(
    test_name: str,
    model: Sequential,
    test_dataset: tf.data.Dataset,
    dataframe: pd.DataFrame
) -> float:
    """
    Executes the behavioral test using the perturbed dataset and returns accuracy.

    Args:
        test_name (str): Name of the test (e.g., 'MFT' or 'INV').
        model (Sequential): Trained TensorFlow/Keras model.
        test_dataset (tf.data.Dataset): Perturbed test dataset.
        dataframe (pd.DataFrame): Corresponding DataFrame to save predictions.

    Returns:
        float: Accuracy of the model on the perturbed dataset.
    """
    try:
        # Extract the text tensors
        for text_batch, _ in test_dataset.take(1):
            text_inputs = text_batch.numpy()

    except Exception as e:
        print(f"[ERROR] Failed to access test dataset: {e}")
        return 0.0

    else:
        predictions = model.predict(text_inputs)
        predicted_labels = np.argmax(predictions, axis=1)

        dataframe["predicted_labels"] = predicted_labels
        dataframe["predicted_probabilities"] = predictions.tolist()

        # Ensure test_results folder exists
        results_dir = os.path.join(os.getcwd(), "test_results")
        os.makedirs(results_dir, exist_ok=True)

        output_path = os.path.join(results_dir, f"{test_name}_test_results.csv")
        dataframe.to_csv(output_path, index=False)

        accuracy = accuracy_score(dataframe["labels"], dataframe["predicted_labels"])
        print(f"[INFO] {test_name} Test Accuracy: {accuracy:.4f}")
        print(f"[INFO] Results saved to {output_path}")

        return accuracy
