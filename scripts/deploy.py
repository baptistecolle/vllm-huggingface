from huggingface_hub import create_inference_endpoint
import os
from dotenv import load_dotenv
import argparse

VLLM_HF_IMAGE_URL = "ghcr.io/baptistecolle/vllm-huggingface:latest"

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a model to Hugging Face Endpoints")
    
    # Model and deployment settings
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-4B",
                      help="Hugging Face model repository ID")
    parser.add_argument("--name", type=str, default=None,
                      help="Endpoint name (defaults to repo_id basename)")
    parser.add_argument("--framework", type=str, default="pytorch",
                      help="Framework to use")
    parser.add_argument("--task", type=str, default="custom",
                      help="Task type")
    parser.add_argument("--accelerator", type=str, default="gpu",
                      help="Accelerator type")
    parser.add_argument("--vendor", type=str, default="aws",
                      help="Cloud vendor")
    parser.add_argument("--region", type=str, default="us-east-1",
                      help="Region for deployment")
    parser.add_argument("--type", type=str, default="protected",
                      help="Endpoint type")
    parser.add_argument("--instance_size", type=str, default="x1",
                      help="Instance size")
    parser.add_argument("--instance_type", type=str, default="nvidia-l4",
                      help="Instance type")
    parser.add_argument("--min-replica", type=str, default="0",
                        help="Minimum number of replicas")
    parser.add_argument("--max-replica", type=str, default="1",
                        help="Maximum number of replicas")
    parser.add_argument("--scale-to-zero-timeout", type=str, default="60",
                        help="Scale to zero timeout")

    # Environment variables
    # parser.add_argument("--disable_sliding_window", type=str, default="true",
    #                   help="Disable sliding window")
    # parser.add_argument("--max_model_len", type=str, default="2048",
    #                   help="Maximum model length")
    # parser.add_argument("--max_num_batched_tokens", type=str, default="8192",
    #                   help="Maximum number of batched tokens")
    # parser.add_argument("--dtype", type=str, default="bfloat16",
    #                   help="Data type")
    # parser.add_argument("--gpu_memory_utilization", type=str, default="0.98",
    #                   help="GPU memory utilization")
    # parser.add_argument("--quantization", type=str, default="fp8",
    #                   help="Quantization type")
    # parser.add_argument("--use_v2_block_manager", type=str, default="true",
    #                   help="Use V2 block manager")
    # parser.add_argument("--vllm_attention_backend", type=str, default="FLASH_ATTN",
    #                   help="VLLM attention backend")
    parser.add_argument("--trust_remote_code", type=str, default="false",
                      help="Trust remote code")
    
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    
    # Set default name if not provided
    if args.name is None:
        args.name = os.path.basename(args.repo_id).lower()
    
    env_vars = {
        # "DISABLE_SLIDING_WINDOW": args.disable_sliding_window,
        # "MAX_MODEL_LEN": args.max_model_len,
        # "MAX_NUM_BATCHED_TOKENS": args.max_num_batched_tokens,
        # "DTYPE": args.dtype,
        # "GPU_MEMORY_UTILIZATION": args.gpu_memory_utilization,
        # "QUANTIZATION": args.quantization,
        # "USE_V2_BLOCK_MANAGER": args.use_v2_block_manager,
        # "VLLM_ATTENTION_BACKEND": args.vllm_attention_backend,
        # "TRUST_REMOTE_CODE": args.trust_remote_code,
    }

    endpoint = create_inference_endpoint(
        name=args.name,
        repository=args.repo_id,
        framework=args.framework,
        task=args.task,
        accelerator=args.accelerator,
        vendor=args.vendor,
        region=args.region,
        type=args.type,
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        custom_image={
            "health_route": "/health",
            "env": env_vars,
            "url": VLLM_HF_IMAGE_URL,
        },
        min_replicas=args.min_replica,
        max_replicas=args.max_replica,
        scale_to_zero_timeout=args.scale_to_zero_timeout,
        # token=os.getenv("HF_TOKEN"),
    )
    
    print(f"Go to https://ui.endpoints.huggingface.co/{endpoint.namespace}/endpoints/{endpoint.name} to see the endpoint status.")
