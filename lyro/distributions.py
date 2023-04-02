import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .random import RandomKey, hash_json

V = TypeVar("V")


class Distribution(ABC, Generic[V]):
    """Abstract base class for immutable distribution objects."""

    @abstractmethod
    async def sample(self, rng: RandomKey) -> V:
        """
        Draws a random sample. This would ideally be deterministic and
        reproducible based on rng, since we can't guarantee this of APIs, we
        can instead simulate determinism by running under a Memoize or
        MemoizeSqlite interpretation.
        """
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
