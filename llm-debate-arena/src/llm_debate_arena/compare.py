"""Model comparison: send one prompt to many models (puzzles, jokes, a dilemma)."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import Completion, LLMRouter, ProviderError, parse_model

PROMPTS: dict[str, str] = {
    "joke": "Tell a joke for a student on the journey to becoming an expert in LLM Engineering",
    "easy": "You toss 2 coins. One of them is heads. What's the probability the other is tails? "
            "Answer with the probability only.",
    "hard": (
        "On a bookshelf, two volumes of Pushkin stand side by side: the first and the second.\n"
        "The pages of each volume together have a thickness of 2 cm, and each cover is 2 mm thick.\n"
        "A worm gnawed (perpendicular to the pages) from the first page of the first volume to the "
        "last page of the second volume.\nWhat distance did it gnaw through?"
    ),
    "dilemma": (
        "You and a partner are contestants on a game show. You're each taken to separate rooms and "
        "given a choice:\n"
        'Cooperate: Choose "Share" - if both of you choose this, you each win $1,000.\n'
        'Defect: Choose "Steal" - if one steals and the other shares, the stealer gets $2,000 and '
        "the sharer gets nothing.\nIf both steal, you both get nothing.\n"
        "Do you choose to Steal or Share? Pick one."
    ),
}


@dataclass
class ComparisonRow:
    spec: str
    text: str = ""
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    seconds: float = 0.0
    cost_cents: float | None = None


def compare_models(
    prompt: str,
    specs: list[str],
    router: LLMRouter | None = None,
    reasoning_effort: str | None = None,
    with_cost: bool = False,
) -> list[ComparisonRow]:
    """Run ``prompt`` on each model; a failing model yields a row with ``error`` set.

    ``with_cost`` uses litellm (optional extra) to compute per-call cost.
    """
    router = router or LLMRouter()
    rows: list[ComparisonRow] = []
    for spec in specs:
        provider, _ = parse_model(spec)
        try:
            if with_cost:
                from .litellm_tools import complete_with_cost

                c = complete_with_cost(spec, [{"role": "user", "content": prompt}])
                rows.append(ComparisonRow(spec, c.text, None, c.input_tokens, c.output_tokens,
                                          c.seconds, c.cost_cents))
                continue
            kwargs = {}
            if reasoning_effort and provider == "openai":
                kwargs["reasoning_effort"] = reasoning_effort
            c: Completion = router.chat(spec, [{"role": "user", "content": prompt}], **kwargs)
            rows.append(ComparisonRow(spec, c.text, None, c.input_tokens, c.output_tokens, c.seconds))
        except (ProviderError, RuntimeError, ImportError) as exc:
            rows.append(ComparisonRow(spec, error=str(exc)))
    return rows


def format_rows(rows: list[ComparisonRow]) -> str:
    out = []
    for r in rows:
        out.append(f"## {r.spec}")
        if r.error:
            out.append(f"ERROR: {r.error}\n")
            continue
        cost = f", cost {r.cost_cents:.4f} cents" if r.cost_cents is not None else ""
        out.append(f"({r.input_tokens} in / {r.output_tokens} out tokens, {r.seconds:.1f}s{cost})\n")
        out.append(r.text + "\n")
    return "\n".join(out)
