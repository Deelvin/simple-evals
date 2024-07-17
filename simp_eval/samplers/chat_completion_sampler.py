import time
from typing import Any
import os

import openai
from openai import OpenAI

from simp_eval.types import MessageList, SamplerBase

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model_name: str,
        url: str,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        token = os.environ.get("OCTOAI_TOKEN", None)
        assert token is not None, "Set OCTOAI_TOKEN"
        self.client = OpenAI(base_url=url + "/v1", api_key=token)
        self.model_name = model_name
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
