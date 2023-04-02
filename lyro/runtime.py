from . import interpreters
from .distributions import Distribution, V


async def sample(name: str, distribution: Distribution[V]) -> V:
    """Sample from a distribution, subject to reinterpretation."""
    return await interpreters.INTERPRETER.sample(name, distribution)
