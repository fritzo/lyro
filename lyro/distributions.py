import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Literal, TypedDict, TypeVar

from .random import RandomKey, hash_json

logger = logging.getLogger(__name__)

V = TypeVar("V")


class Distribution(ABC, Generic[V]):
    """Abstract base class for immutable distributions over strings."""

    @abstractmethod
    async def sample(self, rng: RandomKey) -> V:
        pass

    def json(self) -> dict:
        """Converts self to json-serializable format."""
        return {"class": type(self).__name__, "dict": self.__dict__}

    def __hash__(self) -> int:
        return hash_json(self.json())


class UniformHash(Distribution[str]):
    """Deterministic distribution for testing."""

    def __init__(self, param: Any = None) -> None:
        super().__init__()
        self.param = param

    async def sample(self, rng: RandomKey) -> str:
        text = json.dumps((self.param, rng.state), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        import openai

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
