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

        b = await lyro.sample(f"b_{i}", ChatGPT(bob))
        alice.append(user(b))
        bob.append(assistant(b))


@pytest.mark.asyncio
async def test_gibbs():
    data = {"b_4": "You've convinced me, tabs are better than spaces."}
    gibbs = Gibbs(alice_bob_model, data)

    sample = await gibbs.sample(num_steps=10)

    assert isinstance(sample, dict)
    assert set(sample) == {
        "a_0",
        "b_0",
        "a_1",
        "b_1",
        "a_2",
        "b_2",
        "a_3",
        "b_3",
        "a_4",
    }
