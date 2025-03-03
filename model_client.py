import json
import fire
import os
import uuid
from typing import Optional

import redis
from redis.client import PubSub
from data_model import (
    ModelRequest,
    GenerationArguments,
    BaseModelRequest,
    InstructModelRequest,
)
from typing import Optional
from util import get_connection


def get_completions(
    text: str,
    model_name: str,
    n_generations: int = 1,
    gen_args: Optional[GenerationArguments] = None,
    verbose=True,
    instruct_format=False,
):
    r = get_connection()
    return_key = str(uuid.uuid4())
    queue_name = os.path.basename(os.path.normpath(model_name))
    pubsub = r.pubsub()
    pubsub.subscribe(return_key)
    if verbose:
        print("subscribed to", return_key)
        print("pushing task to", queue_name)
    if instruct_format:
        model_request = InstructModelRequest(
            generation_args=(
                gen_args.model_dump() if gen_args else GenerationArguments()
            ),
            messages=[{"role": "user", "content": text}],
            num_generations=n_generations,
            return_key=return_key,
        )
    else:
        model_request = BaseModelRequest(
            generation_args=(
                gen_args.model_dump() if gen_args else GenerationArguments()
            ),
            text=text,
            num_generations=n_generations,
            return_key=return_key,
        )
    r.rpush(
        queue_name,
        json.dumps(model_request.model_dump()),
    )

    for message in pubsub.listen():
        if message["type"] == "message":
            response: list[str] = json.loads(message["data"])
            break

    pubsub.unsubscribe(return_key)
    if verbose:
        print("got response from", return_key)
    return response


if __name__ == "__main__":
    fire.Fire(get_completions)
