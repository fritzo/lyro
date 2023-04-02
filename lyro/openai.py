import logging
from typing import Any, Dict, List, Literal, TypedDict

import openai

from .distributions import Distribution
from .random import RandomKey

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def system(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def assistant(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


class ChatGPT(Distribution[str]):
    """
    GPT distribution over chat messages.

    This requires the ``openai`` package and environment variables
    OPENAI_API_ORG and OPENAI_API_KEY.
    """

    def __init__(
        self,
        messages: List[ChatMessage],
        *,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = messages

    async def sample(self, rng: RandomKey) -> str:
        # Adapted from API docs
        # https://platform.openai.com/docs/api-reference/chat/create?lang=python
        request: Dict[str, Any] = dict(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            stream=False,
        )
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens

        raw_response = await openai.ChatCompletion.acreate(**request)
        response: dict = raw_response.to_dict_recursive()
        if any(c["finish_reason"] == "length" for c in response["choices"]):
            logger.warning("Chat was truncated")
        text: str = response["choices"][0]["message"]["content"]
        return text
