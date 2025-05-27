@deploy:
    uv run scripts/deploy.py

@inference endpoint-url:
    uv run scripts/inference.py --endpoint-url $endpoint-url

[working-directory: 'docker']
@build:
    docker build -t vllm-huggingface-inference-endpoint . 

@run:
    docker run --runtime nvidia --gpus all -p 80:80 -e MODEL_PATH="Qwen/Qwen3-4B" vllm-huggingface-inference-endpoint