from typing import Callable

from lyro.interpretations import Trace


class Gibbs:
    trace: Trace | None = None

    def __init__(self, model: Callable) -> None:
        self.model = model

    def init(self) -> Trace:
        with Trace() as self.trace:
            self.model()
        return self.trace

    def step(self) -> Trace:
        assert self.trace is not None
        for name, distribution in self.trace.trace.items():
            raise NotImplementedError("TODO")
        return self.trace
