import hashlib
import json
import struct
from typing import NamedTuple, Tuple


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


class RandomKey(NamedTuple):
    """Immutable random state."""

    head: int = 0
    tail: "RandomKey | None" = None

    def split(self) -> Tuple["RandomKey", "RandomKey"]:
        """
        Splits this random key into two keys, (deeper,incremented).

        To ensure this doesn't grow too deep, prefer to use this as::

            new, rng = rng.split()

        where ``rng`` is kept around and ``new`` is consumed sooner.
        """
        deeper = RandomKey(0, self)
        incremented = RandomKey(self.head + 1, self.tail)
        return deeper, incremented
