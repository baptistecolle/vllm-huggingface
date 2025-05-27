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

1. View/Edit the details in `examples/deploy.py`. It is set up to deploy a HuggingFace Inference Endpoint for the [Phi-3-vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct). Once you have set up the necessary variables, run `python examples/deploy.py`.
2. Go to the link printed by the previous `deploy.py` script to watch the endpoint deployment status and to retrieve the inference base url when finished deploying.
3. Copy this Endpoint Url from step 2 and add the env variable `HF_ENDPOINT_URL` with this copied value. Again, see `.env.example` for how to format.

# Run inference
1. The endpoint you have deployed above is OpenAI API Compatible, meaning you can use the OpenAI library and any other library built to use OpenAI's library with your endpoint. To see an example of how you can call inference using your new endpoint, see `examples/inference.py`.
2. To run the inference, run `python examples/inference.py`.
