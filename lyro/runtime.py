from . import interpreters
from .distributions import Distribution


async def sample(name: str, distribution: Distribution) -> str:
    return await interpreters.INTERPRETER.sample(name, distribution)
