import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from .distributions import Distribution, V
from .random import RandomKey, hash_json


class Interpreter(ABC):
    """Abstract base class for probabilistic program interpreters."""

    base: "Interpreter | None" = None

    @abstractmethod
    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        pass

    def __enter__(self) -> "Interpreter":
        if self.base is not None:
            raise RuntimeError("Interpreters are not reentrant: they cannot be nested")
        self.base = INTERPRETER
        set_interpreter(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self.base is not None
        set_interpreter(self.base)
        self.base = None

    def __add__(self, other: "Interpreter") -> "Interpreter":
        """Stack interpreters, e.g. inner + middle + outer."""
        if other.base is not None:
            raise ValueError
        other.base = self
        return other


class Standard(Interpreter):
    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")
        return await distribution.sample(rng)


class If(Interpreter):
    """Conditional interpretation."""

    def __init__(
        self,
        cond: Callable[[str, Distribution], bool],
        true: Interpreter,
    ):
        super().__init__()
        self.cond = cond
        self.true = true

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        if self.cond(name, distribution):
            return await self.true.sample(name, distribution)
        assert self.base is not None
        return await self.base.sample(name, distribution)


class ThreadRandomKey(Interpreter):
    """Thread random keys through a program."""

    def __init__(self, rng: RandomKey = RandomKey(), force: bool = False) -> None:
        super().__init__()
        self.rng = rng
        self.force = force

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        assert self.base is not None
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
            assert self.base is not None
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
        assert self.base is not None
        value = await self.base.sample(name, distribution, rng)

        # Save for later.
        value_json = json.dumps(value)
        with sqlite3.connect(self.dbname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO key_value VALUES (?, ?)", (key_hash, value_json)
            )
        return value


class Trace(Interpreter):
    """Record a program trace."""

    def __init__(self) -> None:
        super().__init__()
        self.trace: Dict[str, Any] = {}

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        assert self.base is not None
        value: V = await self.base.sample(name, distribution, rng)
        self.trace[name] = value
        return value

    def __enter__(self) -> "Trace":
        super().__enter__()
        return self


class Replay(Interpreter):
    """Replay a trace against a program."""

    def __init__(self, trace: Dict[str, Any]) -> None:
        super().__init__()
        self.trace = trace

    async def sample(
        self, name: str, distribution: Distribution[V], rng: RandomKey | None = None
    ) -> V:
        value: V = self.trace[name]
        return value


INTERPRETER: Interpreter = Standard() + Memoize() + ThreadRandomKey()


def set_interpreter(interpreter: Interpreter) -> None:
    """Sets the global interpreter."""
    global INTERPRETER
    INTERPRETER = interpreter
