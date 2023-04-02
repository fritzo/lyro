import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .distributions import Distribution, RandomKey, hash_json

DATA = os.path.join(os.path.dirname(__file__), "data")


class Interpreter(ABC):
    base: "Interpreter" | None = None

    @abstractmethod
    async def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
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
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")
        return await distribution.sample(rng)


class ThreadRandomKey(Interpreter):
    """Thread random keys through a program."""

    def __init__(self, rng: RandomKey = RandomKey(), force: bool = False) -> None:
        super().__init__()
        self.rng = rng
        self.force = force

    async def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        assert self.base is not None
        if rng is None or self.force:
            rng, self.rng = self.rng.split()
        return await self.base.sample(name, distribution, rng)


class Memoize(Interpreter):
    """Memoize in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[Tuple[Distribution, RandomKey], str] = {}

    async def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")
        key = distribution, rng
        if key not in self.cache:
            assert self.base is not None
            self.cache[key] = await self.base.sample(name, distribution, rng)
        return self.cache[key]


class MemoizeSqlite(Interpreter):
    """Memoize in a sqlite database."""

    def __init__(self, dbname: str = os.path.join(DATA, "memo.db")) -> None:
        super().__init__()

        # Initialize the database.
        self.dbname = os.path.abspath(dbname)
        os.makedirs(os.path.dirname(dbname), exist_ok=True)
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS key_value (
                    key_hash INTEGER PRIMARY KEY,
                    value_json TEXT
                )
                """
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.dbname)
        conn.row_factory = sqlite3.Row
        return conn

    async def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")

        # Try to reuse old result.
        key_hash = hash_json((distribution.json(), rng.state))
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM key_value WHERE key_hash = ?", (key_hash,)
            )
            row = cursor.fetchone()
            if row is not None:
                value: str = json.loads(row[0])
                return value

        # Compute new result.
        assert self.base is not None
        value = await self.base.sample(name, distribution, rng)

        # Save for later.
        value_json = json.dumps(value)
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO key_value VALUES (?, ?)", (key_hash, value_json)
            )
        return value


class Trace(Interpreter):
    """Record a program trace."""

    def __init__(self) -> None:
        super().__init__()
        self.trace: Dict[str, str] = {}

    async def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        assert self.base is not None
        value = await self.base.sample(name, distribution, rng)
        self.trace[name] = value
        return value

    def __enter__(self) -> "Trace":
        super().__enter__()
        return self


class Replay(Interpreter):
    """Replay a trace against a program."""

    def __init__(self, trace: Dict[str, str]) -> None:
        self.trace = trace

    async def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        return self.trace[name]


INTERPRETER: Interpreter = Standard() + Memoize() + ThreadRandomKey()


def set_interpreter(interpreter: Interpreter) -> None:
    global INTERPRETER
    INTERPRETER = interpreter
