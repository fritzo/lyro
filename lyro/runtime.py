from . import interpreters
from .distributions import Distribution, V
from .interpreters import Condition


async def sample(
    name: str, distribution: Distribution[V], *, obs: V | None = None
) -> V:
    """Sample from a distribution, subject to reinterpretation."""
    if obs is None:
        return await interpreters.INTERPRETER.sample(name, distribution)
    with Condition({name: obs}):
        return await interpreters.INTERPRETER.sample(name, distribution)
