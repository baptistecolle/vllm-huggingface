from openai import OpenAI
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
    
    # Initialize the client but point it to TGI
    # client = OpenAI(
    #     base_url=args.endpoint_url + "/v1/",  # Add /v1/ to the endpoint URL
    #     api_key=os.getenv("HF_TOKEN")
    # )
    
    client = InferenceClient(base_url=args.endpoint_url, token=os.getenv("HF_TOKEN"))
    
    messages = [
        {"role": "user", "content": "How are you?"}
    ]

    start = time()
    
    # response = client.chat.completions.create(
    #     model="/repository",  # needs to be /repository since there are the model artifacts stored
    #     messages=messages,
    #     max_tokens=30,
    #     temperature=0.0,
    #     stream=False,
    # ).choices[0].message.content
    
    response = client.chat_completion(
        model="/repository",  # needs to be /repository since there are the model artifacts stored
        messages=messages,
        max_tokens=30,
        temperature=0.0,
        stream=False,
    ).choices[0].message.content
    
    print(f"Response: {response}")
    print(f"Time taken: {time() - start:.2f}s")

if __name__ == "__main__":
    main()
