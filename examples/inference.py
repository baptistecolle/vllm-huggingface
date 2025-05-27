from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from time import time
from multiprocessing.pool import ThreadPool


load_dotenv() 
ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")  # if endpoint object is not available check the UI 
API_KEY = os.getenv("HF_TOKEN")
STREAM = False

# initialize the client
client = InferenceClient(model=ENDPOINT_URL, token=API_KEY)


def predict(messages):
    return client.chat_completion(
        messages=messages,
        max_tokens=30,
        temperature=0.0,
        stream=False,
    ).choices[0].message.content

if __name__ == "__main__":
    
    batch_size = 64
    pool = ThreadPool(batch_size)

    messages = [
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920"
                }
            },
            {
                "type": "text",
                "text": "What is in the above image? Explain in detail."
            }
        ]},
    ]

    start = time()
    responses = pool.map(predict, [messages] * batch_size)
    print(responses[0])
    print(f"Time taken: {time() - start:.2f}s")

    start = time()
    responses = list(map(predict, [messages] * batch_size))
    print(f"Time taken: {time() - start:.2f}s")
