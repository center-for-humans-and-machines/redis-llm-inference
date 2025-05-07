import json
import os
import torch
import fire
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from pydantic import BaseModel
import redis


def get_connection() -> redis.Redis:
    """Get a connection to the redis server"""
    host = os.getenv("REDIS_HOST", "localhost")
    print("Connecting redis to", host)
    return redis.Redis(
        host=host,
        port=os.getenv("REDIS_PORT", 6379),
        password=os.getenv("REDIS_PASSWORD", None),
    )


class Message(BaseModel):
    role: str
    content: str


class InstructModelRequest(BaseModel):
    return_key: str
    messages: list[Message]


@torch.inference_mode()
def serve_model(
    model_path: str,
):
    """
    Minimal implementation for serving a full LLM model to process InstructModelRequests from a Redis queue.

    Args:
        model_path: Path to the model to load
        max_context_len: Maximum context length for the model
    """
    # Normalize and expand the model path
    model_path = os.path.normpath(os.path.expanduser(model_path))

    # Use the model directory name as the queue name
    queue_name = os.path.basename(model_path)

    # Connect to Redis
    redis_conn = get_connection()

    # Clear any existing items in the queue
    redis_conn.delete(queue_name)

    # Load the model and tokenizer
    print(f"Loading model from {model_path}...")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Successfully connected to Redis and now listening for tasks on queue: {queue_name}"
    )

    # Process tasks from the queue
    while True:
        # Wait for and get the next item from the queue
        _, json_bytes = redis_conn.blpop(queue_name)
        data_dict = json.loads(json_bytes)

        # Parse the request
        request = InstructModelRequest.model_validate(data_dict)
        print(f"Processing task {request.return_key}")

        # Tokenize the input
        tokenized_input = tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to("cuda")

        tokenized_input = tokenized_input.unsqueeze(0)

        # Generate output
        outputs = model.generate(
            input_ids=tokenized_input,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=128,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )

        # Extract and decode the generated text
        generation = outputs[0, tokenized_input.size(1) :].cpu()
        response = tokenizer.decode(generation, skip_special_tokens=True)

        # Publish the response
        redis_conn.publish(request.return_key, json.dumps(response))
        print(f"Published response to {request.return_key}")


if __name__ == "__main__":
    fire.Fire(serve_model)
