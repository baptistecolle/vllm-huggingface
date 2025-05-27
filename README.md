# VLLM container for HuggingFace Inference Endpoints

This repo publish a VLLM container that is compatible with HuggingFace Inference Endpoints.

# Installation

```bash
uv sync
```

```bash
cp .env.example .env
```

# Deploy to HuggingFace Endpoint

```bash
uv run scripts/deploy.py
```

# Run inference

```bash
uv run scripts/inference.py --endpoint-url <endpoint-url>
```