import os
import requests
from typing import Optional
import argparse


def make_request(url: str, json_data: Optional[dict] = None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RELAY_API_KEY')}",
    }
    response = requests.request("POST", url, headers=headers, json=json_data)
    return response.json()


def main(prompt: str, model_name: str):
    request = [{"role": "user", "content": prompt}]
    model_name = os.path.basename(model_name)
    response = make_request(
        f"{os.getenv('REDIS_RELAY')}/{model_name}",
        {"messages": request},
    )
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    main(args.prompt, args.model_name)
