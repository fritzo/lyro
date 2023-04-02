import hashlib
import sys
from abc import ABC, abstractmethod
from typing import Hashable, Tuple


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
    @abstractmethod
    def sample(self, rng: RandomKey) -> str:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass
