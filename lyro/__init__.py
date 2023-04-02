from __future__ import annotations

import abc
import hashlib
import sys
from typing import Callable, Dict, Hashable, Tuple, TypeVar

T = TypeVar("T", bound="Distribution")


# Distributions.


def hash_int(s: str, num_bits: int = sys.hash_info.width) -> int:
    """Hashes a string into a fixed width int."""

    # Hash the input string using SHA-256
    sha256_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()

    # Convert the hexadecimal hash to an integer
    int_hash = int(sha256_hash, 16)

    # Reduce the integer hash to a 52-bit integer
    reduced_hash: int = int_hash % (2**num_bits)

    return reduced_hash


class RandomKey:
    """Immutable random state."""

    def __init__(self, state: Tuple[Hashable, int] = (None, 0)):
        self.state = state

    def __hash__(self, num_bits: int = sys.hash_info.width) -> int:
        return hash_int(str(self.state), num_bits)

    def split(self) -> Tuple[RandomKey, RandomKey]:
        """
        Split the random state into (longer,shorter). Prefer to use this as::

            new, rng = rng.split()
        """
        left = self.state, 0  # one tuple deeper
        right = self.state[0], self.state[-1]  # same depth
        return RandomKey(left), RandomKey(right)


class Distribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, rng: RandomKey) -> str:
        pass


# Interpreters.


class Interpreter(abc.ABC):
    base: Interpreter | None = None

    @abc.abstractmethod
    def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        pass

    def __enter__(self) -> Interpreter:
        if self.base is not None:
            raise RuntimeError("Interpreters are not reentrant: they cannot be nested")
        self.base = INTERPRETER
        set_interpreter(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self.base is not None
        set_interpreter(self.base)
        self.base = None

    def __add__(self, other: Interpreter) -> Interpreter:
        """Stack interpreters, e.g. inner + middle + outer."""
        if other.base is not None:
            raise ValueError
        other.base = self
        return other


class Standard(Interpreter):
    def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")
        return distribution.sample(rng)


class ThreadRandomKey(Interpreter):
    def __init__(self, rng: RandomKey = RandomKey(), force: bool = False) -> None:
        super().__init__()
        self.rng = rng
        self.force = force

    def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        assert self.base is not None
        if rng is None or self.force:
            rng, self.rng = self.rng.split()
        return self.base.sample(name, distribution, rng)


class Memoize(Interpreter):
    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[Tuple[Distribution, RandomKey], str] = {}

    def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        if rng is None:
            raise ValueError("Missing rng, try adding a ThreadRandomKey")
        key = distribution, rng
        if key not in self.cache:
            assert self.base is not None
            self.cache[key] = self.base.sample(name, distribution, rng)
        return self.cache[key]


class Trace(Interpreter):
    def __init__(self) -> None:
        super().__init__()
        self.trace: Dict[str, str] = {}

    def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        assert self.base is not None
        value = self.base.sample(name, distribution, rng)
        self.trace[name] = value
        return value

    def __enter__(self) -> Trace:
        super().__enter__()
        return self


class Replay(Interpreter):
    def __init__(self, trace: Dict[str, str]) -> None:
        self.trace = trace

    def sample(
        self, name: str, distribution: Distribution, rng: RandomKey | None = None
    ) -> str:
        return self.trace[name]


# Runtime.

INTERPRETER: Interpreter = Standard() + Memoize() + ThreadRandomKey()


def set_interpreter(interpreter: Interpreter) -> None:
    global INTERPRETER
    INTERPRETER = interpreter


def sample(name: str, distribution: Distribution) -> str:
    return INTERPRETER.sample(name, distribution)


# Inference.


class GibbsSampler:
    trace: Trace | None = None

    def __init__(self, model: Callable) -> None:
        self.model = model

    def init(self) -> Trace:
        with Trace() as self.trace:
            self.model()
        return self.trace

    def step(self) -> Trace:
        assert self.trace is not None
        for name, distribution in self.trace.trace.items():
            raise NotImplementedError("TODO")
        return self.trace
