FROM vllm/vllm-openai:latest

# Use a custom entrypoint to align with HuggingFace Inference Endpoints
COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
