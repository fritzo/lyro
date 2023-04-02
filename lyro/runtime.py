from . import interpreters
from .distributions import Distribution, V


async def sample(name: str, distribution: Distribution[V]) -> V:
    return await interpreters.INTERPRETER.sample(name, distribution)
