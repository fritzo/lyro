import hashlib
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, List, Literal, Tuple, TypedDict

logger = logging.getLogger(__name__)


def hash_int(s: str, num_bits: int = sys.hash_info.width) -> int:
    """Deterministically hashes a string into a fixed width int."""
    sha256_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
    int_hash = int(sha256_hash, 16)
    reduced_hash: int = int_hash % (2**num_bits)
    return reduced_hash


def hash_json(data, num_bits: int = sys.hash_info.width) -> int:
    """Deterministically hashes data into a fixed width int."""
    data_json = json.dumps(data, sort_keys=True)
    int_hash: int = hash_int(data_json)
    return int_hash


class RandomKey:
    """Immutable random state."""

    def __init__(self, state: Tuple[Hashable, int] = (None, 0)):
        self.state = state

    def __hash__(self, num_bits: int = sys.hash_info.width) -> int:
        """
        This is the main interface to consume randomness: ``hash(rng)``.

        Warning: hash(...) should be called only once!
        """
        return hash_json(self.state, num_bits)

    def split(self) -> Tuple["RandomKey", "RandomKey"]:
        """
        Splits this random key into two keys, (deeper,shallower).

        To ensure this doesn't grow too deep, prefer to use this as::

            new, rng = rng.split()

        where ``rng`` is kept around and ``new`` is consumed sooner.
        """
        left = self.state, 0  # one tuple deeper
        right = self.state[0], self.state[-1]  # same depth
        return RandomKey(left), RandomKey(right)


class Distribution(ABC, Hashable):
    """Abstract base class for immutable distributions over strings."""

    @abstractmethod
    async def sample(self, rng: RandomKey) -> str:
        pass

    def json(self) -> dict:
        """Converts self to json-serializable format."""
        return {"class": type(self).__name__, "dict": self.__dict__}

    def __hash__(self) -> int:
        return hash_json(self.json())


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChat(Distribution):
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
