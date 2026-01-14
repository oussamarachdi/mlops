FROM python:3.10-slim

WORKDIR /mlflow
RUN pip install mlflow==2.14.1 boto3 cryptography pymysql
