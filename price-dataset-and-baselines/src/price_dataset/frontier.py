"""Zero-shot frontier-model price predictors via litellm.

These models were compared on the first 200 test items (Gemini 3 Pro: 50, 2 workers).
Model ids drift quickly - override with ``--model`` (any litellm model string).
"""

from __future__ import annotations

# litellm model id -> (extra kwargs). Models used in the reference comparison.
FRONTIER_MODELS: dict[str, dict] = {
    "openai/gpt-4.1-nano": {},
    "anthropic/claude-opus-4-5": {},
    "gemini/gemini-3-pro-preview": {"reasoning_effort": "low"},
    "gemini/gemini-3.1-flash-lite": {},
    "xai/grok-4-1-fast-non-reasoning": {"seed": 42},
    "gpt-5.1": {"reasoning_effort": "high", "seed": 42},
}


def messages_for(item_or_text) -> list[dict]:
    text = getattr(item_or_text, "summary", item_or_text)
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{text}"
    return [{"role": "user", "content": message}]


def make_frontier_pricer(model: str, completion_fn=None, **extra):
    """Return predictor(item) -> raw reply string (the evaluator parses the number)."""
    if not extra:
        extra = dict(FRONTIER_MODELS.get(model, {}))

    def predictor(item):
        completion = completion_fn
        if completion is None:
            from litellm import completion  # lazy
        response = completion(model=model, messages=messages_for(item), **extra)
        return response.choices[0].message.content

    predictor.__name__ = model.split("/")[-1].replace("-", "_").replace(".", "__")
    return predictor
