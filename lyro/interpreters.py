import functools
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Generic, NamedTuple, TypeVar

from .distributions import Distribution, V
from .random import RandomKey, hash_json

T = TypeVar("T")


class Interpreter(ABC):
    """Abstract base class for probabilistic program interpreters."""

    base: "Interpreter"

    @abstractmethod
    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        pass

    def __enter__(self) -> "Interpreter":
        self.base = INTERPRETER
        set_interpreter(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        set_interpreter(self.base)

    def __call__(self, model: Callable[[], Awaitable[T]]) -> "Decorator[T]":
        return Decorator(self, model)

    def __add__(self, other: "Interpreter") -> "Interpreter":
        """Stack interpreters, e.g. inner + middle + outer."""
        other.base = self
        return other


class Decorator(Generic[T]):
    def __init__(self, interpreter: Interpreter, model: Callable[[], Awaitable[T]]):
        super().__init__()
        self.interpreter = interpreter
        self.model = model
        functools.update_wrapper(self, model)

    async def __call__(self) -> T:
        with self.interpreter:
            return await self.model()


class Standard(Interpreter):
    async def sample(
        self,
        name: str,
        distribution: Distribution[V],
        rng: RandomKey | None = None,
    ) -> V:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")
        return await distribution.sample(rng)


# Create a default standard base interpreter, whose base is itself.
BASE = Standard()
Interpreter.base = BASE
assert BASE.base is BASE  # BASE all the way down.


class ThreadRandomKey(Interpreter):
    """Thread random keys through a program."""

    def __init__(self, rng: RandomKey = RandomKey(), force: bool = False) -> None:
        super().__init__()
        self.rng = rng
        self.force = force

    async def sample(
        self,
        name: str,
        distribution: Distribution[V],
        rng: RandomKey | None = None,
    ) -> V:
        if rng is None or self.force:
            rng, self.rng = self.rng.split()
        return await self.base.sample(name, distribution, rng)


class Memoize(Interpreter):
    """Memoize in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[int, Any] = {}

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")

        key = hash((distribution, rng))
        if key not in self.cache:
            self.cache[key] = await self.base.sample(name, distribution, rng)
        value: V = self.cache[key]
        return value


class MemoizeSqlite(Interpreter):
    """Memoize in a sqlite database."""

    def __init__(self, dbname: str) -> None:
        super().__init__()

        # Initialize the database.
        self.dbname = os.path.abspath(dbname)
        os.makedirs(os.path.dirname(self.dbname), exist_ok=True)
        with sqlite3.connect(self.dbname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS key_value (
                    key_hash INTEGER PRIMARY KEY,
                    value_json TEXT
                )
                """
            )

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")

        # Try to reuse old result.
        key = distribution.json(), rng
        key_hash = hash_json(key)
        print("DEBUG", key, key_hash)
        with sqlite3.connect(self.dbname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value_json FROM key_value WHERE key_hash = ?", (key_hash,)
            )
            row = cursor.fetchone()
            if row is not None:
                value: V = json.loads(row[0])
                return value

        # Compute new result.
        value = await self.base.sample(name, distribution, rng)

        # Save for later.
        value_json = json.dumps(value)
        with sqlite3.connect(self.dbname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO key_value VALUES (?, ?)", (key_hash, value_json)
            )
        return value


class TraceNode(NamedTuple):
    name: str
    distribution: Distribution
    rng: RandomKey | None
    value: Any


class Trace(Interpreter):
    """Record program execution in a trace."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: Dict[str, TraceNode] = {}

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        value: V = await self.base.sample(name, distribution, rng)
        self.nodes[name] = TraceNode(name, distribution, rng, value)
        return value

    def __enter__(self) -> "Trace":
        super().__enter__()
        return self


class Condition(Interpreter):
    """Condition a program on data."""

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__()
        assert isinstance(data, dict)
        self.data = data

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        try:
            value: V = self.data[name]
        except KeyError:
            value = await self.base.sample(name, distribution, rng)
        return value


class If(Interpreter):
    """Conditional interpretation."""

    def __init__(
        self, cond: Callable[[str, Distribution], bool], true: Interpreter
    ) -> None:
        super().__init__()
        self.cond = cond
        self.true = true

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        if self.cond(name, distribution):
            return await self.true.sample(name, distribution, rng)
        return await self.base.sample(name, distribution, rng)


INTERPRETER: Interpreter = BASE + Memoize() + ThreadRandomKey()


def set_interpreter(interpreter: Interpreter = BASE) -> None:
    """Sets the global interpreter."""
    global INTERPRETER
    INTERPRETER = interpreter
