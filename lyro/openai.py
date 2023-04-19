import copy
import logging
import textwrap
from typing import Any, Dict, List, Literal, TypedDict

import openai

from .distributions import Distribution
from .random import RandomKey

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


Chat = List[ChatMessage]


def system(content: str) -> ChatMessage:
    """Helper to construct a system message."""
    return ChatMessage(role="system", content=content)


def user(content: str) -> ChatMessage:
    """Helper to construct a user message."""
    return ChatMessage(role="user", content=content)


def assistant(content: str) -> ChatMessage:
    """Helper to construct an assistant message."""
    return ChatMessage(role="assistant", content=content)


class ChatGPT(Distribution[str]):
    """
    GPT distribution over chat messages, conditioned on chat history.

    This requires the ``openai`` package and environment variables
    OPENAI_API_ORG and OPENAI_API_KEY.
    """

    def __init__(
        self,
        messages: List[ChatMessage],
        *,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.messages = copy.deepcopy(messages)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def sample(self, rng: RandomKey) -> str:
        # Form a request.
        # https://platform.openai.com/docs/api-reference/chat/create?lang=python
        request: Dict[str, Any] = dict(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
        )
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens

        # Call the async non-streaming API.
        raw_response = await openai.ChatCompletion.acreate(**request)

        # Parse the response.
        response: dict = raw_response.to_dict_recursive()
        if any(c["finish_reason"] == "length" for c in response["choices"]):
            logger.warning("Chat was truncated")
        text: str = response["choices"][0]["message"]["content"]
        return text


SYSTEM_PROMPT = """You are Hercule Pyrot, the brilliant detective who has listened in to the conversations of multiple speakers."""


def render_messages(messages: Chat) -> str:
    lines: List[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        lines.append(f"{role.capitalize()}:")
        for line in textwrap.wrap(content, width=76):
            lines.append(f"    {line}")
    return "\n".join(lines)


class FusedGPT(Distribution[str]):
    """
    Corresponds to the complete conditional probabiltic model::

        x = lyro.sample("x", ChatGPT(prior))
        for i, (pos, likelihood, obs) in enumerate(likelihoods):
            likelihood[pos] = x
            lyro.sample(f"y{i}", likelihood[:-1], obs=likelihood[-1])

    which is compiled from a user-facing program like::

        def model():
            x = lyro.sample("x", ChatGPT(prior))
            y1 = lyro.sample("y1", ChatGPT(prompt1 + [x]))
            y2 = lyro.sample("y2", ChatGPT(prompt2 + [x] + [y1]))

        infer(model, data={"y1": y1, "y2": y2})
    """

    VARIABLE = "MYSTERY_UTTERANCE"

    def __init__(self, prior: Chat, likelihoods: List[Chat]) -> None:
        super().__init__()
        self.prior = copy.deepcopy(prior)
        self.likelihoods = copy.deepcopy(likelihoods)

    async def sample(self, rng: RandomKey) -> str:
        assert self.prior[-1]["role"] == "user", self.prior[-1]["role"]
        messages = self.prior + [assistant("MYSTERY_UTTERANCE")]
        question = f"""The conversation starts like this:

{render_messages(messages)}

Now after the MYSTERY_UTTERANCE you heard {len(self.likelihoods)} side conversations."""
        for messages in self.likelihoods:
            question += f"""

One side conversation went like this:

{render_messages(messages)}
"""
        question += """Your task now, Hercule Pyrot, is to guess MYSTERY_UTTERANCE. Please write your best guess below, exactly as it would have appeared in the conversation."""
        messages = [system(SYSTEM_PROMPT), user(question)]
        logger.info("DEBUG:\n{messages}")
        fused = ChatGPT(messages)
        return await fused.sample(rng)
