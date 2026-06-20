# 🚀 Enterprise-Grade LLMOps Pipeline & Advanced RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-v1.28%2B-326CE5.svg)](https://kubernetes.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-0194E2.svg)](https://mlflow.org/)

A production-ready, highly optimized LLMOps serving and fine-tuning pipeline engineered to deliver low-latency, cost-effective inference for millions of concurrent users. This repository demonstrates end-to-end machine learning engineering (MLE) practices including multi-stage containerization, Kubernetes orchestration, Prometheus observability, GitOps CI/CD, model compression, and advanced Retrieval-Augmented Generation (RAG) architectures.

---

## 📞 Recruiter & Hiring Manager Contact Info
* **Name:** Indrasena Reddy Bathini
* **Email:** indrasenareddybathini86@gmail.com
* **Phone:** +91 9392698146
* **LinkedIn:** [Indrasena Reddy Bathini](https://www.linkedin.com/in/indra-reddy-b-52603b282/)
* **GitHub Portfolio:** [indrareddy12](https://github.com/indrareddy12)
* **Availability:** Immediate / Open to Opportunities
* **Desired Roles:** Machine Learning Engineer (MLE), LLMOps Engineer, MLOps Engineer, Senior Software Engineer (AI/ML)

---

## 🏗️ Production Architecture & System Design

This pipeline is designed for horizontal scalability, zero-downtime deployments, and resilient performance under load.

```mermaid
graph TD
    Client[Client Apps / Users] -->|HTTPS Requests| ALB[AWS ALB / Ingress Controller]
    ALB -->|Load Balancing| K8sCluster[Kubernetes Pods: FastAPI Servicers]
    
    subgraph FastAPI Pods [FastAPI / Uvicorn Service Pods]
        Server[FastAPI Inference Engine]
        Cache[Local Redis Cache / Prompt Cache]
        ModelLoader[In-Memory Model: DistilGPT-2 / Quantized LLM]
        PromExporter["/metrics Endpoint"]
        
        Server <--> Cache
        Server --> ModelLoader
        Server --> PromExporter
    end
    
    K8sCluster --> FastAPI Pods
    
    subgraph Vector DB Layer [Retrieval-Augmented Generation]
        VectorDB[(Vector DB: Pinecone / Milvus / Qdrant)]
        SparseDB[(Sparse Search: BM25 / ElasticSearch)]
    end
    
    Server -->|Hybrid Search Query| VectorDB
    Server -->|Lexical Match Query| SparseDB
    
    subgraph Experiment & Model Registry
        MLflow[MLflow Tracking Server]
        S3Bucket[(S3 Model Artifacts Store)]
        Registry[MLflow Model Registry]
    end
    
    ModelLoader -->|Pull Registered Model| Registry
    Registry <--> S3Bucket
    MLflow <--> Registry

    subgraph Observability
        Prometheus[Prometheus Server]
        Grafana[Grafana Dashboards]
    end
    
    Prometheus -->|Scrape metrics| PromExporter
    Grafana -->|Query Dashboard Data| Prometheus
```

---

## ⚡ High-Concurrency Serving & Latency Optimization

To serve millions of users while maintaining sub-second latency (P95 < 200ms), the system implements several state-of-the-art serving optimizations:

1. **Continuous & Dynamic Batching:** Groups incoming concurrent request tokens dynamically to maximize GPU core utilization and increase throughput by up to 4x compared to naive sequential queuing.
2. **Key-Value (KV) Caching:** Caches key-value attention pairs for generated tokens in GPU memory, avoiding redundant computation during autoregressive generation cycles.
3. **Model Quantization (AWQ/GPTQ 4-bit):** Compresses model weights from FP16 to INT4, lowering memory footprint by ~75% and enabling larger context sizes and faster token generation speeds with minimal degradation in perplexity.
4. **Asynchronous Runtime:** Built on Python's `asyncio` loop with **FastAPI** and multi-worker **Gunicorn** (`UvicornWorker`), allowing non-blocking I/O operations for concurrent health checks, tracing, and metric collection.

---

## 🔍 Advanced Retrieval-Augmented Generation (RAG) Architecture

Beyond basic vector lookups, the system supports advanced search workflows designed to mitigate hallucinations and deliver contextually rich context windows:

* **Semantic & Parent-Child Chunking:** Segments source documents into parent paragraphs with smaller child chunks for precise vector embedding matches, retaining broad contextual integrity for generation.
* **Dense + Sparse Hybrid Search:** Combines dense semantic vector retrieval (e.g., via BGE/Cohere embeddings) with classic BM25 lexical keyword matching, merged via Reciprocal Rank Fusion (RRF).
* **Cross-Encoder Re-ranking:** Utilizes a lightweight cross-encoder model to re-evaluate the top 50 retrieved documents, prioritizing the absolute most relevant context in the final 5-10 context slots.
* **Context Compression & Query Expansion:** Automatically rewrites ambiguous user queries using a small distilled LLM step and filters redundant tokens in retrieved documents to conserve token count and cost.
* **Self-RAG Guardrails:** Validates retrieved documents for topic relevance before feeding them into the LLM, and evaluates generated answers for faithfulness against the retrieved facts (minimizing hallucination risk).

---

## 💰 Production Improvement & Cost Optimization

Serving large models at scale can be prohibitively expensive. This pipeline implements proactive cloud cost controls:

* **Model Distillation & Task Specialization:** Fine-tunes specialized open-source models (such as `distilgpt2` or small Llama variants) using Hugging Face `Trainer` and MLflow. This enables high task performance on specific workloads at a fraction of the hardware cost of general-purpose APIs (like GPT-4).
* **Prompt Caching:** Intercepts identical system prompts and retrieval prefixes at the gateway level, eliminating downstream LLM processing costs for repetitive inputs.
* **Kubernetes HPA (Horizontal Pod Autoscaler):** Automatically scales inference pods up and down based on target CPU utilization (60%) and custom GPU occupancy metrics. This minimizes idle compute resources during low-traffic periods.
* **Spot Instance Resiliency:** Supports container teardowns gracefully by utilizing Kubernetes pre-stop lifecycles, enabling the use of cheap cloud Spot instances with zero disruption to active user requests.

---

## 📊 Observability & Experiment Tracking

* **Real-time Monitoring:** Exposes a Prometheus `/metrics` endpoint tracking request count (`llm_requests_total`) and generation latency (`llm_request_latency_seconds`) segmented by endpoints.
* **Experiment Management:** Integrates with MLflow to track hyperparameters, datasets, loss curves, and validation metrics during model fine-tuning.
* **Model Registry:** Organizes and transitions trained models through stages (`Staging`, `Production`, `Archived`), facilitating automated canary deployments and rollbacks.

---

## 🛠️ Project Structure

```
├── ci-cd/
│   └── github-actions.yaml      # Automated GitHub Actions lint, test, build, push, deploy
├── deployment/
│   ├── Dockerfile               # Multi-stage optimized production Docker configuration
│   └── k8s-deployment.yaml      # Kubernetes manifests (Deployment, ClusterIP Service, HPA)
├── models/
│   └── fine-tune.py             # Hugging Face Trainer + MLflow experiment tracking script
├── monitoring/
│   └── mlflow-tracking.py       # Helper script to launch local MLflow tracking server
├── scripts/
│   ├── ecr-push.sh              # Template script to authenticate and push images to AWS ECR
│   └── sagemaker-deploy.sh      # Template for packaging models and deploying to AWS SageMaker
├── model_server.py              # FastAPI high-performance inference server with Prometheus metrics
├── entrypoint.sh                # Optimized container start script using Gunicorn + Uvicorn
├── requirements.txt             # Project library dependencies
└── pytest_example_test.py       # Smoke tests for CI pipeline verification
```

---

## 🚀 Quickstart Guide

### A. Local Development

1. **Clone the Repository & Enter Workspace:**
   ```bash
   git clone https://github.com/indrareddy12/LLMOps-Pipeline.git
   cd LLMOps-Pipeline
   ```

2. **Setup Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on macOS/Linux:
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Start the MLflow Tracking Server (Optional):**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

4. **Run the FastAPI Server locally:**
   ```bash
   export MODEL_NAME=distilgpt2  # Use lightweight model for local execution
   export MLFLOW_TRACKING_URI=http://localhost:5000
   uvicorn model_server:app --host 0.0.0.0 --port 8000 --log-level info
   ```

5. **Test Inference & Metrics Endpoints:**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Send inference prompt
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"input":"Deep learning in production requires","max_new_tokens":25}'

   # Fetch Prometheus metric dump
   curl http://localhost:8000/metrics
   ```

---

### B. Containerization & Docker Build

The Dockerfile is structured using **multi-stage builds** to minimize runtime image size (slimming build dependencies) and implements running as a non-privileged user (`appuser`) for enhanced container security.

1. **Build the production Docker Image:**
   ```bash
   docker build -t llmops-pipeline:latest -f deployment/Dockerfile .
   ```

2. **Run container locally with mapped ports:**
   ```bash
   docker run --rm -p 8000:8000 \
     -e MODEL_NAME=distilgpt2 \
     -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
     --name llmops-service llmops-pipeline:latest
   ```

---

### C. Kubernetes Orchestration

Deploy a scalable, load-balanced pod topology with configured liveness/readiness probes and horizontal auto-scaling:

1. **Set Up Kubernetes (Minikube / EKS / GKE):**
   ```bash
   # Build the image directly inside Minikube's Docker daemon if testing locally:
   eval $(minikube docker-env)
   docker build -t llmops-pipeline:latest -f deployment/Dockerfile .
   eval $(minikube docker-env -u)
   ```

2. **Apply Kubernetes Manifests:**
   ```bash
   kubectl apply -f deployment/k8s-deployment.yaml
   ```

3. **Verify Deployment & Autoscaler (HPA):**
   ```bash
   kubectl get pods -w
   kubectl get service llmops-service
   kubectl get hpa
   ```

4. **Port-Forward to Access Service Locally:**
   ```bash
   kubectl port-forward svc/llmops-service 8000:80
   ```

---

### D. Model Fine-Tuning Pipeline

Train and evaluate custom models utilizing Hugging Face Trainer and track all artifacts to the central MLflow registry.

1. **Trigger Fine-Tuning Execution:**
   ```bash
   python models/fine-tune.py
   ```
2. **Review Metrics and Registry:**
   Open `http://localhost:5000` to review epoch steps, validation metrics, and download trained parameters.

---

### E. GitHub Actions CI/CD Pipeline

The repository includes a fully configured workflow under `ci-cd/github-actions.yaml` which automatically runs upon any push to `main` or pull request:
1. **Lint & Test:** Runs code checks and executes unit/integration tests with `pytest`.
2. **Docker Build & Push:** Compiles the multi-stage production container and pushes it securely to GitHub Container Registry (`ghcr.io`).
3. **Deploy:** Automatically updates target Kubernetes deployments via `kubectl apply` utilizing secret credentials (`KUBE_CONFIG`).
