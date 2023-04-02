from typing import Awaitable, Callable

from lyro.interpretations import Trace


class Gibbs:
    state: Trace | None = {}

    def __init__(self, model: Callable[[], Awaitable]) -> None:
        self.model = model

    async def init(self) -> Trace:
        with Trace() as self.trace:
            await self.model()
        return self.trace

    async def step(self) -> Trace:
        assert self.trace is not None
        for name, node in self.trace.nodes.items():
            raise NotImplementedError("TODO")
        return self.trace
