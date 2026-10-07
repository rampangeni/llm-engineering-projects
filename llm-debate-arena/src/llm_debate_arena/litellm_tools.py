"""litellm helpers: unified ``completion`` + cost reporting + prompt-caching demo.

litellm is an optional dependency: ``pip install 'llm-debate-arena[litellm]'``.
litellm model names use ``provider/model`` (e.g. ``gemini/gemini-3.1-flash-lite``);
we accept our ``provider:model`` specs too and convert them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from importlib import resources

from dotenv import load_dotenv

from .llm import Completion


def to_litellm_name(spec: str) -> str:
    """``anthropic:claude-haiku-4-5`` -> ``anthropic/claude-haiku-4-5``; bare names -> ``openai/...``."""
    from .llm import PROVIDERS, parse_model

    provider, model = parse_model(spec)
    if "/" in spec and ":" not in spec.split("/")[0]:
        return spec  # already litellm style
    return f"{'ollama_chat' if provider == 'ollama' else provider}/{model}"


def _completion():
    try:
        from litellm import completion
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "litellm is not installed. Run: pip install 'llm-debate-arena[litellm]'"
        ) from exc
    load_dotenv(override=True)
    return completion


class CostedCompletion(Completion):
    pass


def complete_with_cost(spec: str, messages: list[dict[str, str]], completion_fn: Callable | None = None):
    """Call litellm; returns a Completion with an extra ``cost_cents`` attribute."""
    fn = completion_fn or _completion()
    start = time.perf_counter()
    resp = fn(model=to_litellm_name(spec), messages=messages)
    hidden = getattr(resp, "_hidden_params", {}) or {}
    cost = hidden.get("response_cost")
    c = CostedCompletion(
        spec=spec,
        text=resp.choices[0].message.content or "",
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        seconds=time.perf_counter() - start,
    )
    c.cost_cents = None if cost is None else cost * 100
    return c


def load_hamlet() -> str:
    return resources.files("llm_debate_arena").joinpath("data/hamlet.txt").read_text(encoding="utf-8")


CACHE_QUESTION = "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?"


def cache_demo(model: str = "gemini/gemini-3.1-flash-lite", completion_fn: Callable | None = None) -> list[dict]:
    """Reproduce the prompt-caching experiment.

    1. Ask without context (cheap, model answers from memory).
    2. Ask with the whole play appended (large input, may not be cached yet).
    3. Repeat step 2 - providers with implicit caching (Gemini, OpenAI) bill the repeated
       prefix as cached tokens at a discount. Returns one stats dict per step.
    """
    fn = completion_fn or _completion()
    hamlet = load_hamlet()
    steps = [("no context", CACHE_QUESTION),
             ("with Hamlet (1st)", CACHE_QUESTION + "\n\nFor context, here is the entire text of Hamlet:\n\n" + hamlet),
             ("with Hamlet (2nd, cache hit expected)", CACHE_QUESTION + "\n\nFor context, here is the entire text of Hamlet:\n\n" + hamlet)]
    results = []
    for label, content in steps:
        resp = fn(model=model, messages=[{"role": "user", "content": content}])
        details = getattr(resp.usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        cost = (getattr(resp, "_hidden_params", {}) or {}).get("response_cost")
        results.append({
            "step": label,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "cached_tokens": cached,
            "cost_cents": None if cost is None else cost * 100,
            "answer": (resp.choices[0].message.content or "").strip(),
        })
    return results
