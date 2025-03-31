import json
import os
import threading

import fire
import torch

from uuid import uuid4
import os
import redis
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from typing import Optional
from data_model import (
    BaseModelRequest,
    InstructModelRequest,
    ModelRequest,
    ValueModelRequestBase,
    ValueModelRequestInstruct,
    ValueModelRequest,
)
from util import get_connection
from typing_extensions import TYPE_CHECKING


def register_i_am_here(channel: str):
    r = get_connection()
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    my_id = str(uuid4())
    for message in pubsub.listen():
        if message["type"] == "message":
            print(f"Received message {message['data']} on {channel}")
            r.publish(message["data"], my_id)


def load_maybe_peft_model_tokenizer(
    model_path,
    device_map="auto",
    quantization_config: Optional[BitsAndBytesConfig] = None,
    attention_implementation="eager",
    torch_dtype=torch.bfloat16,
    use_unsloth=False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    extra_args = {}
    tokenizer = None
    if quantization_config is not None:
        extra_args["quantization_config"] = quantization_config
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        if use_unsloth:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=4096,
                dtype=torch_dtype,
                load_in_4bit=quantization_config is not None
                and quantization_config.load_in_4bit,
                device_map=device_map,
            )
            FastLanguageModel.for_inference(model)
        elif os.path.isfile(os.path.join(model_path, "head_configs.json")):
            from transformer_heads import load_lora_with_heads
            from transformer_heads.util.helpers import get_model_params

            model_params = get_model_params(base_model_name)
            model = load_lora_with_heads(
                base_model_class=model_params["model_class"],
                path=model_path,
                quantization_config=quantization_config,
                only_inference=True,
                attn_implementation=attention_implementation,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                attn_implementation=attention_implementation,
                torch_dtype=torch_dtype,
                **extra_args,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            attn_implementation=attention_implementation,
            torch_dtype=torch_dtype,
            **extra_args,
        )
        base_model_name = model_path

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.inference_mode()
def queue_processing(
    model_path: str,
    use_unsloth: bool = False,
    attention_implementation: str = "flash_attention_2",
    max_context_len=4096,
    verbose=True,
):
    model_path = os.path.normpath(os.path.expanduser(model_path))
    device = "cuda"
    key_name = os.path.basename(model_path)

    reg_thread = threading.Thread(target=register_i_am_here, args=(f"{key_name}_here",))
    reg_thread.start()

    r = get_connection()
    r.delete(key_name)
    model, tokenizer = load_maybe_peft_model_tokenizer(
        model_path=model_path,
        use_unsloth=use_unsloth,
        attention_implementation=attention_implementation,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(
        "Successfully connected to redis and now listening for tasks on queue", key_name
    )
    while 1:
        json_bytes = r.blpop(key_name)[1]
        datadic = json.loads(json_bytes)
        match datadic["type"]:
            case "base":
                data = BaseModelRequest.model_validate(datadic)
                tok_text: torch.Tensor = tokenizer.encode(
                    data.text, return_tensors="pt"
                ).to(device)
                if verbose:
                    print("Got new task", data.return_key, data.text[-30:])
            case "instruct":
                data = InstructModelRequest.model_validate(datadic)
                tok_text = tokenizer.apply_chat_template(
                    data.messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True,
                ).to(device)
                if verbose:
                    print("Got new task", data.return_key, data.messages[-1])
            case "value_base":
                data = ValueModelRequestBase.model_validate(datadic)
                tok_text = tokenizer.encode(data.text, return_tensors="pt").to(device)
                if verbose:
                    print("Got new task", data.return_key, data.text[-30:])
            case "value_instruct":
                data = ValueModelRequestInstruct.model_validate(datadic)
                tok_text = tokenizer.apply_chat_template(
                    data.messages,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    tokenize=True,
                )
                if verbose:
                    print("Got new task", data.return_key, data.messages[-1])

        if isinstance(data, ModelRequest):
            if len(tok_text) > max_context_len - data.generation_args.max_new_tokens:
                tok_text = tok_text[
                    -max_context_len + data.generation_args.max_new_tokens :
                ]
            tok_text = tok_text.repeat(int(data.num_generations), 1)

            outputs = model.generate(
                input_ids=tok_text,
                generation_config=data.generation_args.to_generation_config(),
                pad_token_id=tokenizer.pad_token_id,
            )
            generations = outputs[:, tok_text.size(1) :].cpu()
            sequence_lengths = (
                torch.eq(generations, tokenizer.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths = sequence_lengths % generations.shape[-1]
            sequence_lengths = sequence_lengths

            response = [
                tokenizer.decode(g[: l + 1], skip_special_tokens=True)
                for g, l in zip(generations, sequence_lengths)
            ]
        elif isinstance(data, ValueModelRequest):
            if TYPE_CHECKING:
                from transformer_heads.output import HeadedModelOutput

            output: HeadedModelOutput = model(tok_text)
            response = float(output.preds_by_head[data.head_name][0, -1])
        else:
            raise ValueError(f"Unknown data type {data}")

        r.publish(data.return_key, json.dumps(response))
        print(
            f"published response {response}",
            data.return_key,
        )


if __name__ == "__main__":
    fire.Fire(queue_processing)
