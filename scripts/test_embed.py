# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import time
from itertools import chain

import requests
from PIL import Image
import torch
from openai import OpenAI

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"
# openai_api_base = "https://kino.lmms-lab.com/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

def run_embed_image():
    start_time = time.perf_counter()
    response = requests.post(
        "http://localhost:30000/v1/embeddings",
        json={
            "model":
            model,
            "messages": [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Represent the given image."
                    },
                ],
            }],
            "encoding_format":
            "float",
        },
    )
    print(f"Generating Image embedding ...")
    response.raise_for_status()
    response_json = response.json()
    data = torch.tensor(response_json["data"][0]["embedding"])
    print(f"Embedding shape: {data.shape}")
    end_time = time.perf_counter()
    used_time = end_time - start_time
    print(f"Using : {used_time:.4f}")

    # print("Embedding output:", response_json["data"][0]["embedding"])

def run_embed_audio():
    # audio_url = AudioAsset("winning_call").url
    audio_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/winning_call.ogg"
    start_time = time.perf_counter()

    response = requests.post(
        "http://localhost:30000/v1/embeddings",
        json={
            "model":
            f"{model}",
            "messages": [{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?"
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            # Any format supported by librosa is supported
                            "url": audio_url
                        },
                    },
                ],
            }],
            "encoding_format":
            "float",
        },
    )
    print(f"Generating Audio embedding ...")
    response.raise_for_status()
    response_json = response.json()
    data = torch.tensor(response_json["data"][0]["embedding"])
    print(f"Embedding shape: {data.shape}")
    end_time = time.perf_counter()
    used_time = end_time - start_time
    print(f"Using : {used_time:.4f}")


def run_openai_batch():
    print("Generating with batched text")
    text = ["What is the captial of France", "The capital of France is Paris"]
    text = text * 5
    start_time = time.perf_counter()
    response = client.embeddings.create(
        model=model, input=text, encoding_format="float"
    )
    end_time = time.perf_counter()

    data = [d.embedding for d in response.data]
    data = torch.tensor(data)
    print(f"Embedding shape: {data.shape}")
    used_time = end_time - start_time
    print(f"Using : {used_time:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Script to call a specified VLM through the API. Make sure to serve "
        "the model with --task embed before running this.")
    parser.add_argument("--model",
                        type=str,
                        required=False,
                        help="Which model to call.")
    args = parser.parse_args()
    run_embed_image()
    run_embed_audio()
    run_openai_batch()
