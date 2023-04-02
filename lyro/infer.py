import asyncio
import itertools
import logging
from collections import Counter
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Tuple

import lyro

from .distributions import Distribution
from .interpreters import Trace

logger = logging.getLogger(__name__)


class Gibbs:
    """Gibbs sampling over static program structure."""

    trace: Trace
    markov_blanket: Dict[str, List[str]]

    def __init__(
        self,
        model: Callable[[], Awaitable],
        *,
        sleep_interval: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.tasks: Dict[str, asyncio.Task] = {}
        self.counts: Counter[str] = Counter()
        self.sleep_interval = sleep_interval

    async def init(self) -> None:
        """Initializes inference. Must be called before :meth:`step`."""
        # Sample from the prior.
        with Trace() as self.trace:
            await self.model()
        nodes = self.trace.nodes

        # Track dependencies.
        # FIXME we need to exclude observed nodes.
        # FIXME this is a simple complete blanket dependency.
        self.markov_blanket = {name: list(nodes) for name in nodes}

        # We use reverse rank to quickly recover from the initial trace that
        # is biased towards the prior and ignores observations.
        self.rank = {name: i for i, name in enumerate(reversed(nodes))}

    async def _find_work(self) -> str:
        # Find currently feasible tasks.
        while True:
            feasible = [
                name
                for name, deps in self.markov_blanket.items()
                if name not in self.tasks
                if not any(dep in self.tasks for dep in deps)
            ]
            if feasible:
                break
            await asyncio.sleep(self.sleep_interval)  # allow work to complete

        # Find the best task to perform, based on previous execution count.
        best = min(feasible, key=lambda name: (self.counts[name], self.rank[name]))
        self.counts[best] += 1
        return best

    async def _do_work(self, name: str) -> None:
        logger.debug(f"Gibbs step at site {repr(name)}")
        node = self.trace.nodes[name]
        local_posterior: Distribution = node.distribution  # FIXME compute posterior
        with Trace() as trace:
            await lyro.sample(name, local_posterior)
        self.trace.nodes.update(trace.nodes)
        self.tasks.pop(name)

    async def step(self) -> str:
        """Runs a single inference step, at one sample site."""

        try:
            self.trace
        except AttributeError:
            raise RuntimeError("Called .step() before .init()")

        name = await self._find_work()
        assert name not in self.tasks
        self.tasks[name] = asyncio.create_task(self._do_work(name))
        return name

    async def run(
        self, num_steps: int | None = None
    ) -> AsyncGenerator[Tuple[int, str], None]:
        """
        Run inference for given number of steps or forever.

        Yields a tuple (step, name) where step is the step number and name is
        the name of the site that is resampled.
        """
        try:
            await self.init()
            for i in range(num_steps) if num_steps else itertools.count():
                name = await self.step()
                yield i, name
        except asyncio.CancelledError:
            for task in self.tasks.values():
                task.cancel()
            self.tasks.clear()
            raise
        await asyncio.gather(*self.tasks.values())
