import asyncio
import itertools
import logging
from collections import Counter
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Tuple

import lyro

from .distributions import Distribution
from .interpreters import Condition, Trace

logger = logging.getLogger(__name__)


class Gibbs:
    """Gibbs sampling over static program structure."""

    trace: Trace
    markov_blanket: Dict[str, List[str]]

    def __init__(
        self,
        model: Callable[[], Awaitable],
        data: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.data = data
        self.tasks: Dict[str, asyncio.Task] = {}
        self.counts: Counter[str] = Counter()

    async def init(self) -> None:
        """Initializes inference. Must be called before :meth:`step`."""
        logger.debug("Gibbs initializing")
        # Sample from the prior.
        with Condition(self.data), Trace() as self.trace:
            await self.model()

        # Validate data.
        assert set(self.data) <= set(self.trace.nodes)

        # Restrict to latent variables.
        nodes = {
            name: node
            for name, node in self.trace.nodes.items()
            if name not in self.data
        }

        # Track dependencies.
        # FIXME this is a simple complete blanket dependency.
        self.markov_blanket = {name: list(nodes) for name in nodes}

        # We use reverse rank to quickly recover from the initial trace that
        # is biased towards the prior and ignores observations.
        self.rank = {name: i for i, name in enumerate(reversed(nodes))}

    async def _find_work(self) -> str:
        # Find currently feasible tasks.
        while True:
            slowest = min(self.counts.values()) if self.counts else 0
            feasible = [
                name
                for name, deps in self.markov_blanket.items()
                if name not in self.tasks  # don't duplicate work
                if not any(dep in self.tasks for dep in deps)  # avoid conflict
                if self.counts[name] <= slowest + 1  # don't get too far ahead
            ]
            if feasible:
                break
            # Wait for more work to complete.
            await asyncio.wait([asyncio.sleep(0)])

        # Find the best task to perform, based on previous execution count.
        best = min(feasible, key=lambda name: (self.counts[name], self.rank[name]))
        self.counts[best] += 1
        return best

    async def _do_work(self, name: str) -> None:
        logger.debug(f"Gibbs step at site {repr(name)}")
        node = self.trace.nodes[name]

        # Compute local posterior.
        prior = node.distribution
        local_posterior: Distribution = prior  # FIXME compute posterior

        try:
            with Trace() as trace:
                await lyro.sample(name, local_posterior)
            self.trace.nodes.update(trace.nodes)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(e)
            raise
        finally:
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
        Run inference for given number of steps or until cancelled.

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

    async def sample(self, num_steps: int) -> Dict[str, Any]:
        """Sample latent variables."""
        # Run inference.
        async for _ in self.run(num_steps):
            pass

        # Validate final state.
        assert all(
            self.trace.nodes[name].value == value for name, value in self.data.items()
        )

        # Return posterior samples of latent variables.
        return {
            name: node.value
            for name, node in self.trace.nodes.items()
            if name not in self.data
        }
