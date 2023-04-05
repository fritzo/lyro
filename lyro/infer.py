import asyncio
import logging
from collections import Counter
from typing import Any, Awaitable, Callable, Dict, List

import lyro

from .distributions import Distribution
from .interpreters import Condition, Trace

logger = logging.getLogger(__name__)


class Gibbs:
    """
    Gibbs sampling over static program structure.

    Args:
        model: A probabilistic model with lyro.sample statements but
            no observe statements.
        data: Observed data on which to condition the model.
    """

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
        self.counts: Counter[str] = Counter()  # #completed inference steps per site
        self.num_pending = 0

    async def _init(self) -> None:
        """Initializes inference.."""
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

    def _find_work(self) -> str | None:
        # Find all currently feasible tasks.
        slowest = min(self.counts.values()) if self.counts else 0
        feasible = [
            name
            for name, deps in self.markov_blanket.items()
            if name not in self.tasks  # don't duplicate work
            if not any(dep in self.tasks for dep in deps)  # avoid conflict
            if self.counts[name] <= slowest  # don't get too far ahead
        ]
        if not feasible:
            return None

        # Find the best task to perform, based on previous execution count.
        best = min(feasible, key=lambda name: (self.counts[name], self.rank[name]))
        self.counts[best] += 1
        return best

    def _start_work(self) -> None:
        """Starts as much work as possible."""
        while self.num_pending:
            name = self._find_work()
            if name is None:
                return
            assert name not in self.tasks
            self.num_pending -= 1
            self.tasks[name] = asyncio.create_task(self._do_work(name))

    async def _do_work(self, name: str) -> None:
        logger.debug(f"Gibbs step at site {repr(name)}")
        node = self.trace.nodes[name]

        # Compute local posterior.
        prior = node.distribution
        local_posterior: Distribution = prior  # FIXME compute posterior

        # Draw a local sample.
        try:
            # FIXME shouldn't we call the model here?
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

        # Check for more work.
        self._start_work()

    async def sample(self, num_steps: int) -> Dict[str, Any]:
        """Draw a single sample of latent variables, running inference."""
        assert isinstance(num_steps, int) and num_steps >= 0

        # Run inference.
        self.num_pending = num_steps
        try:
            await self._init()
            self._start_work()
        except asyncio.CancelledError:
            self.num_pending = 0
            for task in self.tasks.values():
                task.cancel()
            self.tasks.clear()
            raise
        while self.tasks:
            await asyncio.gather(*self.tasks.values())
        assert self.num_pending == 0

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
