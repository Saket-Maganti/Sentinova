# Sentinova: MLOps-Driven Sentiment Analysis

Sentinova is an end-to-end sentiment analysis pipeline powered by MLOps. It combines modern NLP techniques with robust workflow orchestration, continuous integration, and cloud-native deployment.

## 🔍 Overview

- **Deep Learning Model**: Bidirectional LSTM with BERT tokenizer for robust sentiment classification.
- **MLOps Workflow**: Automated pipelines using Airflow, Docker, MLflow, and GitHub Actions.
- **Cloud Native**: Model deployment via AWS SageMaker using production-ready Docker images.
- **Benchmarking**: Uses `Checklist` to behaviorally test model robustness with negation and invariance tests.

## 🔧 Tech Stack

- Python, TensorFlow, Keras, Transformers (BERT)
- Apache Airflow (workflow orchestration)
- MLflow (experiment tracking & model registry)
- AWS (EC2, S3, ECR, SageMaker)
- Snowflake (data warehouse)
- Docker + GitHub Actions (CI/CD)


## 🚀 Features

- **ETL DAG**: Scrapes tweets via `snscrape`, labels with polarity, and loads to Snowflake
- **Model DAG**: Trains BiLSTM inside GPU-accelerated container with BERT tokenization
- **MLflow**: Tracks training runs and pushes models to registry
- **Benchmarking**: Evaluates robustness with negation & invariance tests using `Checklist`
- **CI/CD**: Deploys production model to SageMaker via GitHub Actions and Amazon ECR

## 📈 Behavioral Testing

Model testing beyond accuracy:

- **MFT**: Negation-based test (e.g., "I like this" → "I don't like this")
- **INV**: Typo/contraction test (e.g., "I can't wait" → "I can not wait")

Both tests are automated and logged into test_results.

## 🧪 Model

- BiLSTM with BERT tokenizer
- Categorical cross-entropy loss
- MLflow logs all training metadata, parameters, and artifacts

## 📦 Deployment

- Docker image built with model + dependencies → pushed to **AWS ECR**
- Deployment to **AWS SageMaker** using MLflow’s `sagemaker._deploy()` function
- Endpoint ready for inference via REST


---


📚 Inspired by: [Checklist for Behavioral Testing] https://github.com/Jithsaavvy/Sentiment-analysis-from-MLOps-paradigm