# LLMOps Pipeline: End-to-End MLOps for LLMs

## Overview
A production-ready pipeline for serving, fine-tuning, and deploying Large Language Models (LLMs) with modern MLOps best practices. Includes:
- FastAPI model server
- MLflow experiment tracking
- Docker & Kubernetes deployment
- GitHub Actions CI/CD
- AWS SageMaker integration (template)
- Prometheus monitoring

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
  - [A. Local Development](#a-local-development)
  - [B. Docker Build & Run](#b-docker-build--run)
  - [C. Kubernetes Deployment](#c-kubernetes-deployment)
  - [D. GitHub Actions CI/CD](#d-github-actions-cicd)
  - [E. AWS SageMaker Deployment (Template)](#e-aws-sagemaker-deployment-template)
  - [F. Fine-tuning Example](#f-fine-tuning-example)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [License](#license)

---

## Features
- **FastAPI** server for LLM inference (`/predict`), health checks (`/health`), and Prometheus metrics (`/metrics`)
- **MLflow** for experiment tracking and model registry
- **Docker** for reproducible builds
- **Kubernetes** manifests for scalable deployment (with HPA, probes, Prometheus annotations)
- **GitHub Actions** for CI/CD (build, test, push, deploy)
- **SageMaker** deployment script (template)
- **Fine-tuning** script with Hugging Face Transformers + MLflow

---

## Prerequisites
- **Git**
- **Python 3.11+** (or 3.10+)
- **pip**
- **Docker** (for containerization)
- **kubectl** + **minikube** (or access to a k8s cluster)
- **(Optional)** AWS CLI & credentials (for ECR/SageMaker)
- **(Optional)** MLflow CLI (`pip install mlflow`)

---

## Project Structure
- `model_server.py` — FastAPI inference server
- `models/fine_tune.py` — Example fine-tuning script (HF Transformers + MLflow)
- `monitoring/mlflow_tracking.py` — MLflow server helpers
- `deployment/dockerfile` — Production Dockerfile
- `deployment/k8s_deployment.yaml` — Kubernetes manifests (Deployment, Service, HPA)
- `ci_cd/github_actions.yaml` — GitHub Actions workflow
- `scripts/` — Helper scripts (e.g., `sagemaker_deploy.sh`)
- `requirements.txt` — Python dependencies
- `entrypoint.sh` — Container entrypoint

---

## Quickstart

### A. Local Development
1. **Clone and enter the project:**
   ```sh
   git clone <repo-url>
   cd llmops-pipeline
   ```
2. **Create & activate virtual environment, install dependencies:**
   ```sh
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **(Optional) Start MLflow server locally:**
   ```sh
   pip install mlflow
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   # In a new terminal:
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```
4. **Set model env var and run the server:**
   ```sh
   export MODEL_NAME=distilgpt2  # or gpt2
   uvicorn model_server:app --host 0.0.0.0 --port 8000 --log-level info
   ```
5. **Test endpoints:**
   ```sh
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"input":"Hello world","max_new_tokens":20}'
   curl http://localhost:8000/metrics
   ```

---

### B. Docker Build & Run
1. **Build the Docker image:**
   ```sh
   docker build -t llmops-pipeline:local -f deployment/dockerfile .
   ```
2. **Run the container:**
   ```sh
   docker run --rm -p 8000:8000 \
     -e MODEL_NAME=distilgpt2 \
     -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
     --name llmops-local llmops-pipeline:local
   ```
3. **Test endpoints as above.**

---

### C. Kubernetes Deployment
1. **Start minikube (if using locally):**
   ```sh
   minikube start --driver=docker
   eval $(minikube docker-env)
   docker build -t llmops-pipeline:local -f deployment/dockerfile .
   eval $(minikube docker-env -u)
   ```
2. **Edit `deployment/k8s_deployment.yaml` to use `llmops-pipeline:local` as the image.**
3. **Apply manifests:**
   ```sh
   kubectl apply -f deployment/k8s_deployment.yaml
   kubectl get pods -w
   kubectl port-forward svc/llmops-service 8000:80
   # Test endpoints at http://localhost:8000/health
   ```

---

### D. GitHub Actions CI/CD
1. **Push code to GitHub:**
   ```sh
   git init
   git add .
   git commit -m "LLMOps pipeline"
   git branch -M main
   git remote add origin git@github.com:<your-org>/<repo>.git
   git push -u origin main
   ```
2. **Configure GitHub Secrets:**
   - `KUBE_CONFIG` (for k8s deploy)
   - Any registry/cloud secrets needed
3. **Check Actions tab:**
   - On push, `.github/workflows/ci_cd/github_actions.yaml` will run: lint, test, build, push, deploy

---

### E. AWS SageMaker Deployment (Template)
1. **Edit and use `scripts/sagemaker_deploy.sh` with your AWS details.**
2. **Run the script:**
   ```sh
   bash scripts/sagemaker_deploy.sh
   ```
   - Packages and uploads model artifacts to S3
   - Creates SageMaker model and endpoint

---

### F. Fine-tuning Example
1. **Set MLflow tracking URI:**
   ```sh
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```
2. **Run fine-tuning script:**
   ```sh
   python models/fine_tune.py
   ```
   - Training logs and artifacts will appear in MLflow UI

---

## Troubleshooting & Tips
- **Port in use:** Only one process can use a port at a time. Stop other servers or use a different port.
- **Model OOM:** Use `distilgpt2` or a smaller model.
- **MLflow UI not reachable:** Ensure MLflow is running and `MLFLOW_TRACKING_URI` is set.
- **Docker networking:** On Linux, use `--network host` or run MLflow inside the container if `host.docker.internal` is unavailable.
- **Secrets:** Never commit credentials. Use GitHub/Kubernetes/Cursor AI secrets.
- **Long model downloads:** First run downloads model files (hundreds of MB). Use smaller models for iteration.
- **Kubernetes:** Use `kubectl get pods`, `kubectl logs <pod>`, and `kubectl describe` for debugging.

---

## License
[MIT](LICENSE)
