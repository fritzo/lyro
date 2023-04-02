from . import interpreters
from .distributions import Distribution


def sample(name: str, distribution: Distribution) -> str:
    return interpreters.INTERPRETER.sample(name, distribution)
