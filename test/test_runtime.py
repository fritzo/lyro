import logging

import pytest

import lyro
from lyro.distributions import ChatGPT, UniformHash, assistant, system, user
from lyro.interpreters import ThreadRandomKey

logger = logging.getLogger(__name__)


async def hash_model():
    x = "foo"
    for i in range(10):
        x = await lyro.sample(f"x_{i}", UniformHash(x))
    return x


async def alice_bob_model():
    alice = [
        system("You try to persuade the user that tabs are better than spaces."),
        user("Which is better, tabs or spaces?"),
    ]
    bob = [
        system("You try to pursuade the user that spaces are better than tabs."),
    ]

    for i in range(5):
        a = await lyro.sample(f"a_{i}", ChatGPT(alice, max_tokens=100))
        alice.append(assistant(a))
        bob.append(user(a))
        logger.info(f"Alice: {a}")

        b = await lyro.sample(f"b_{i}", ChatGPT(alice, max_tokens=100))
        alice.append(user(b))
        bob.append(assistant(b))
        logger.info(f"Bob: {b}")

    return alice, bob


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [hash_model, alice_bob_model])
async def test_chatgpt_diversity(model):
    x = await model()
    y = await model()
    assert x != y


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [hash_model, alice_bob_model])
async def test_chatgpt_caching(model):
    with ThreadRandomKey():
        x = await model()
    with ThreadRandomKey():
        y = await model()
    assert x == y
