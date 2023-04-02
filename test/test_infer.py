import logging

import pytest

import lyro
from lyro.infer import Gibbs
from lyro.openai import ChatGPT, assistant, system, user

logger = logging.getLogger(__name__)


async def alice_bob_model(num_steps: int = 5):
    alice = [
        system(
            "You are role playing Alice, who is trying to convince Bob to "
            "use tabs rather than spaces. You respond with a single brief "
            "persuasive sentence. "
        ),
        user("Spaces are more canonical than tabs."),
    ]
    bob = [
        system(
            "You are role playing Bob, who is trying to convince Alice to "
            "use spaces rather than tabs. You respond with a single brief "
            "persuasive sentence. "
        ),
    ]

    for i in range(num_steps):
        a = await lyro.sample(f"a_{i}", ChatGPT(alice))
        alice.append(assistant(a))
        bob.append(user(a))

        b = await lyro.sample(f"b_{i}", ChatGPT(alice))
        alice.append(user(b))
        bob.append(assistant(b))


@pytest.mark.asyncio
async def test_gibbs():
    model = lyro.Condition(
        {
            "b_4": "You've convinced me, tabs are better than spaces.",
        }
    )(alice_bob_model)

    gibbs = Gibbs(model)
    async for i, name in gibbs.run(num_steps=10):
        logger.info(f"step {i}: {name}")
