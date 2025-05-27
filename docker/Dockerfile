FROM vllm/vllm-openai:v0.8.5.post1

# Use a custom entrypoint to align with HuggingFace Inference Endpoints
COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
