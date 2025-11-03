"""Helpers to start and configure MLflow tracking server locally (template).
In production, run a managed MLflow server with backend DB and artifact store.
"""
import os
def start_local_mlflow():
    # Example commands to run externally:
    # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    print('To start MLflow locally run:')
    print('mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000')
if __name__ == '__main__':
    start_local_mlflow()