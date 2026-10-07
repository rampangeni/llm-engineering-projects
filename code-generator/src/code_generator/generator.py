"""LLM call: Python -> C++/Rust, streamed."""

from __future__ import annotations

import re
from typing import Callable, Iterator

from openai import OpenAI

from code_generator.models import ModelSpec, find_model, make_client
from code_generator.prompts import messages_for

ClientFactory = Callable[[str], OpenAI]

_FENCE_RE = re.compile(r"```[a-zA-Z+]*\n?")


def strip_fences(text: str) -> str:
    """Remove markdown code fences (rather than chained .replace('```cpp', '') etc.)."""
    return _FENCE_RE.sub("", text).replace("```", "").strip() + "\n"


def stream_port(
    model: str | ModelSpec,
    python: str,
    language: str,
    compile_command: list[str],
    system_info: dict | None = None,
    client_factory: ClientFactory = make_client,
) -> Iterator[str]:
    """Yield the *accumulated* generated code as chunks arrive (fences stripped).

    Gradio re-renders the whole output component per yield, so we yield the
    cumulative text rather than deltas.
    """
    spec = model if isinstance(model, ModelSpec) else find_model(model)
    client = client_factory(spec.provider)
    kwargs = {}
    if spec.reasoning_effort:
        kwargs["reasoning_effort"] = spec.reasoning_effort
    stream = client.chat.completions.create(
        model=spec.name,
        messages=messages_for(python, language, compile_command, system_info),
        stream=True,
        **kwargs,
    )
    acc = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            acc += delta
            yield _FENCE_RE.sub("", acc).replace("```", "")
    yield strip_fences(acc)


def port(*args, **kwargs) -> str:
    """Non-streaming convenience: return the final code."""
    last = ""
    for last in stream_port(*args, **kwargs):
        pass
    return last
