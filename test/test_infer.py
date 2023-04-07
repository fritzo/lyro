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
async def test_gibbs_ab():
    data = {"b_4": "You've convinced me, tabs are better than spaces."}
    gibbs = Gibbs(alice_bob_model, data)

    sample = await gibbs.sample(num_steps=10)

    assert isinstance(sample, dict)
    expected = {"a_0", "b_0", "a_1", "b_1", "a_2", "b_2", "a_3", "b_3", "a_4"}
    assert set(sample) == expected


async def book_model(dynamic: bool = False):
    """
    title <-- book --> summary
                |
                V
              moral
    """
    # Sample the main book text.
    messages = [
        system("You are a writer of children's books."),
        user("Write a children's book about the Pyro team."),
    ]
    book = await lyro.sample("book", ChatGPT(messages))

    # Write a title for the book.
    messages = [
        system("You are an editor of children's books."),
        user("Write a good title for following book:\n\n{book}"),
    ]
    title = await lyro.sample("title", ChatGPT(messages))

    # Summarize the book.
    messages = [
        system("You are a promoter of children's books."),
        user("Write a description about the following book:\n\n{book}"),
    ]
    summary = await lyro.sample("summary", ChatGPT(messages))

    # Extract a moral from the book.
    messages = [
        system("You are a reviewer of children's books."),
        user("Summarize the moral lesson of the following book:\n\n{book}"),
    ]
    moral = await lyro.sample("moral", ChatGPT(messages))

    revised = []
    if dynamic:
        # Sample the images, note this has dynamic structure.
        paragraphs = [line.strip() for line in book.splitlines() if line.strip()]
        for i, paragraph in enumerate(paragraphs):
            messages = [
                system("You are an editor for a children's books."),
                user(f"Revise the following paragraph:\n\n{paragraph}"),
            ]
            revised.append(await lyro.sample(f"revised{i}", ChatGPT(messages)))

    return book, title, summary, moral, revised


@pytest.mark.asyncio
async def test_gibbs_book():
    data = {"title": "Pyro meets its new little sibling, Lyro"}
    markov_blanket = {
        "book": ["title", "summary", "moral"],
        "title": ["book"],
        "summary": ["book"],
        "moral": ["book"],
    }
    gibbs = Gibbs(book_model, data, markov_blanket=markov_blanket)

    sample = await gibbs.sample(num_steps=10)

    assert isinstance(sample, dict)
    expected = {"summary", "book", "moral"}
    assert set(sample) == expected
