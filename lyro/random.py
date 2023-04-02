import hashlib
import json
import struct
from typing import Hashable, Tuple


def hash_text(text: str) -> int:
    """Deterministically hashes a string into a fixed width int."""
    sha256 = hashlib.sha256(text.encode("utf-8"))
    hash_bytes = sha256.digest()
    hash_int: int = struct.unpack("<q", hash_bytes[:8])[0]
    return hash_int


def hash_json(data) -> int:
    """Deterministically hashes data into a fixed width int."""
    data_json = json.dumps(data, sort_keys=True)
    int_hash: int = hash_text(data_json)
    return int_hash


class RandomKey:
    """Immutable random state."""

    def __init__(self, state: Tuple[Hashable, int] = (None, 0)):
        self.state = state

    def __hash__(self) -> int:
        """
        This is the main interface to consume randomness: ``hash(rng)``.

        Warning: hash(...) should be called only once!
        """
        return hash_json(self.state)

    def split(self) -> Tuple["RandomKey", "RandomKey"]:
        """
        Splits this random key into two keys, (deeper,shallower).

        To ensure this doesn't grow too deep, prefer to use this as::

            new, rng = rng.split()

        where ``rng`` is kept around and ``new`` is consumed sooner.
        """
        left = self.state, 0  # one tuple deeper
        right = self.state[0], self.state[-1] + 1  # same depth
        return RandomKey(left), RandomKey(right)
