#!/usr/bin/env python3

"""
Deploy a production-ready MLflow model from EC2 to AWS SageMaker.

Steps performed:
1. Load production model from MLflow registry (hosted on EC2).
2. Package the model into a Docker image and push to AWS ECR.
3. Deploy the image from ECR to AWS SageMaker.
4. Create an endpoint to enable real-time inference.

Author: Jithin Sasikumar | Modified by Vaishnavi Polampalli
"""

import os
import sys
import mlflow
from mlflow import sagemaker

# Add project root to sys.path for importing local utilities
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config

# Load configuration
config = Config()

# Set MLflow tracking URI (remote EC2-based MLflow server)
mlflow.set_tracking_uri(config["model-tracking"]["mlflow_tracking_uri"])

# Endpoint name in SageMaker
app_name = config["model-deploy"]["endpoint_name"]

# Name of model in MLflow registry and its URI (ensure it's in "Production" stage)
model_name = config["model-registry"]["model_name"]
model_uri = f"models:/{model_name}/production"

# Fetch deployment environment variables
try:
    docker_image_url = os.environ["IMAGE_URI"]
    iam_role_arn = os.environ["ARN_ROLE"]
    aws_region = os.environ["REGION"]
except KeyError as e:
    print(f"[ERROR] Required environment variable not found: {e}")
    sys.exit(1)

# Deploy model to SageMaker
try:
    print(f"[INFO] Deploying model '{model_name}' to SageMaker endpoint '{app_name}'...")
    sagemaker._deploy(
        mode='create',
        app_name=app_name,
        model_uri=model_uri,
        image_url=docker_image_url,
        execution_role_arn=iam_role_arn,
        instance_type='ml.m5.xlarge',
        instance_count=1,
        region_name=aws_region
    )
    print(f"[SUCCESS] Model deployed successfully. Endpoint name: {app_name}")

except Exception as e:
    print(f"[ERROR] Deployment failed: {e}")
    sys.exit(1)
