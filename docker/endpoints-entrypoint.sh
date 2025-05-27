# Set TGI like environment variables
NUM_SHARD=${NUM_SHARD:-$(nvidia-smi --list-gpus | wc -l)}
# the weights are stored in the /repository directory as specify at https://huggingface.co/docs/inference-endpoints/guides/custom_container
MODEL_PATH=${MODEL_PATH:-"/repository"}

# Entrypoint for the OpenAI API server
CMD="vllm serve $MODEL_PATH --host '0.0.0.0' --port 80 --tensor-parallel-size '$NUM_SHARD' --enable-auto-tool-choice --tool-call-parser hermes --served-model-name /repository data-agents/data-agents-qwen3-4b"

# Execute the command
eval $CMD
