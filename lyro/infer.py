from typing import Awaitable, Callable

from lyro.interpretations import Trace


class Gibbs:
    trace: Trace | None = None

    def __init__(self, model: Callable[[], Awaitable]) -> None:
        self.model = model

    async def init(self) -> Trace:
        with Trace() as self.trace:
            await self.model()
        return self.trace

    async def step(self) -> Trace:
        assert self.trace is not None
        for name, distribution in self.trace.trace.items():
            raise NotImplementedError("TODO")
        return self.trace
