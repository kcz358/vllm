import base64

import requests
from openai import OpenAI, AsyncOpenAI
import asyncio
from copy import deepcopy
import time

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"
# openai_api_base = "https://kino.lmms-lab.com/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

async_client = AsyncOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result
# Video input inference
def run_video() -> None:
    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
    video_base64 = encode_base64_content_from_url(video_url)

    messages = [{
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": "Please describe this video in detail"
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{video_base64}"
                    },
                },
            ],
        }]

    # Suppose 1 hr video, the video url is 1 min long
    batched_messages = [deepcopy(messages) for _ in range(60)]


    res = []
    async def run():
        sem = asyncio.Semaphore(24)
        async def _process(messages):
            async with sem:
                return await async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                )

        all_task = [asyncio.create_task(_process(messages)) for messages in batched_messages]
        for task in all_task:
            result = await task
            res.append(result.choices[0].message.content)
    
    start_time = time.perf_counter()
    asyncio.run(run())
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time}")

if __name__ == "__main__":
    run_video()