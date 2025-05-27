from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from time import time
import argparse

load_dotenv() 

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with Hugging Face endpoint')
    parser.add_argument('--endpoint-url', type=str, required=True,
                      help='Hugging Face endpoint URL')
    return parser.parse_args()

def main():
    
    args = parse_args()

    client = InferenceClient(model=args.endpoint_url)
    messages = [
        {"role": "user", "content": "How are you?"}
    ]

    start = time()
    
    response = client.chat_completion(
        messages=messages,
        max_tokens=30,
        temperature=0.0,
        stream=False,
    ).choices[0].message.content
    
    print(f"Response: {response}")
    print(f"Time taken: {time() - start:.2f}s")


if __name__ == "__main__":
    main()
