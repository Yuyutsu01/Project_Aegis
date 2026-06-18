# Deployment Guide

This guide covers deployment procedures for local Docker Compose development and production Kubernetes environments.

## 1. Local Compose Deployment

To build and run all services locally (FastAPI API gateway, Streamlit Dashboard, and Prometheus monitoring):

```bash
cd deployment/docker/
docker compose up --build
```

Access dashboards:
- Streamlit Quant Terminal: `http://localhost:8501`
- Prometheus Dashboard: `http://localhost:9090`

To skip retraining pipelines and load precompiled models directly:
```bash
SKIP_PIPELINE=true docker compose up
```

## 2. Kubernetes Cloud Deployment

To deploy to a production Kubernetes cluster (AWS EKS or GCP GKE):

1. **Verify ConfigMaps:**
   Apply configurations in `configs/prod.yaml` as Kubernetes Secret/ConfigMaps.
2. **Apply Manifests:**
   ```bash
   kubectl apply -f deployment/kubernetes/deployments/
   kubectl apply -f deployment/kubernetes/services/
   ```
3. **Set Secrets:**
   Ensure database passwords and broker credentials (`PROD_DB_PASSWORD`, `FYERS_ACCESS_TOKEN`) are configured using Kubernetes Secrets.
