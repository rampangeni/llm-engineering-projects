# LLM Debate Arena

A toolkit for talking to many LLM providers through one interface:
- **`run`** - a configurable multi-bot conversation (the classic argumentative-GPT vs polite-Claude chat, generalised to N bots from any providers).
- **`compare`** - send one prompt (a joke, the coin puzzle, the Pushkin-worm puzzle, the prisoner's dilemma, or your own) to many models and see answers, tokens and latency (and cost via litellm).
- **`cache-demo`** - reproduce the prompt-caching experiment: ask about Hamlet with and without the full play in the prompt and watch cached tokens/cost.
- **Gradio app** - conversation tab (live transcript) and model-comparison tab.

**What you will learn:** frontier model APIs (OpenAI-compatible endpoints for Anthropic/Gemini/DeepSeek/Groq/Grok/OpenRouter/Ollama, reasoning effort, litellm, prompt caching with `hamlet.txt`, adversarial conversation).

## Architecture (summary)
`cli/app` -> `debate` (engine, transcript-to-messages) / `compare` -> `llm` (provider registry + router over the OpenAI SDK) and `litellm_tools` (optional). Details: [docs/lld-design.md](docs/lld-design.md).

## Prerequisites
- Python >= 3.11.
- API keys for the providers you use (all optional, see `.env.example`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `GROK_API_KEY`, `OPENROUTER_API_KEY`.
- Free/local option: [Ollama](https://ollama.com) (`ollama serve`, `ollama pull llama3.2`). Use the `local` preset. No GPU strictly required.
- `cache-demo` and `--cost` need the `litellm` extra and a Gemini (default) key.

## Setup
```bash
cp .env.example .env
uv venv && uv pip install -e ".[dev,litellm]"      # drop ,litellm if you don't need cost/caching
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,litellm]"
```

## Run
```bash
debate-arena presets
debate-arena run --preset classic --rounds 5            # GPT-4.1-mini vs Claude Haiku 4.5
debate-arena run --preset local                         # all on Ollama
debate-arena run --preset roundtable --topic "Is Python still the best language for AI?"
debate-arena run --config my_debate.json
debate-arena compare dilemma -m openai:gpt-5-nano -m anthropic:claude-sonnet-4-5-20250929 -m ollama:llama3.2 --reasoning-effort minimal
debate-arena compare "Describe the colour blue to someone blind in one sentence" -m gemini:gemini-2.5-flash --cost
debate-arena cache-demo --model gemini/gemini-3.1-flash-lite
debate-arena ui
```
Model spec: `provider:model` where provider is one of `openai, anthropic, gemini, deepseek, groq, grok, openrouter, ollama` (no prefix = OpenAI).

### Config file
```json
{
  "rounds": 4,
  "topic": "Tabs vs spaces",
  "bots": [
    {"name": "Ann", "model": "openai:gpt-4.1-mini", "system_prompt": "You love tabs and are fiercely loyal."},
    {"name": "Bob", "model": "ollama:llama3.2", "system_prompt": "You prefer spaces; be witty."}
  ],
  "seed": [{"speaker": "Ann", "content": "Hi there"}, {"speaker": "Bob", "content": "Hi"}]
}
```

## Tests
`pytest` - offline; LLM router, litellm and HTTP are faked.

## Troubleshooting
- `X_API_KEY is not set`: add it to `.env` (the error names the variable).
- Anthropic 400 about message order: the engine merges same-role messages and starts with a user message; custom configs should too if you bypass the engine.
- Model not found: provider model ids change often; pass the current id.
- `cache-demo` shows `cached=0` on the second call: implicit caching is best-effort and has minimum prompt sizes; retry shortly after.
- Cost: the hamlet prompt is ~60k tokens per call (3 calls in the demo) - cheap on flash-lite models, not free.
