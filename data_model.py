from pydantic import BaseModel, Field
from typing import Optional


class GenerationArguments(BaseModel):
    max_new_tokens: int = Field(default=128)
    min_new_tokens: Optional[int] = None

    # Generation strategy
    do_sample: bool = Field(default=True)

    # Hyperparameters for logit manipulation
    temperature: float = Field(default=1.0)
    top_k: Optional[int] = Field(default=0)
    top_p: float = Field(default=1.0)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def to_generation_config(self):
        from transformers import GenerationConfig

        return GenerationConfig(**self.model_dump())


class ModelRequest(BaseModel):
    return_key: str
    num_generations: int
    generation_args: GenerationArguments = Field(default_factory=GenerationArguments)


class BaseModelRequest(ModelRequest):
    text: str
    type: str = "base"


class Message(BaseModel):
    role: str
    content: str


class InstructModelRequest(ModelRequest):
    messages: list[Message]
    type: str = "instruct"


class ValueModelRequest(BaseModel):
    return_key: str
    head_name: str = "value_head"


class ValueModelRequestBase(ValueModelRequest):
    text: str
    type: str = "value_base"


class ValueModelRequestInstruct(ValueModelRequest):
    messages: list[Message]
    type: str = "value_instruct"
