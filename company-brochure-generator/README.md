# Company Brochure Generator

Give it a company name and landing-page URL. It (1) asks an LLM which links on the page matter for a brochure (About, Careers, Customers...) and returns them as JSON, (2) scrapes those pages, and (3) streams a markdown brochure for prospective customers, investors and recruits. Pick the model/provider (GPT, Claude, Gemini, Ollama) and tone.

**What you will learn:** a two-step "link selection -> brochure" pipeline, one-shot JSON prompting, streaming output, and a Gradio streaming UI with a model dropdown.

## Features
- Two-step multi-call LLM pattern (a first "agentic" pattern).
- JSON link selection with robust parsing + relative-URL resolution.
- Streaming output in CLI and Gradio.
- Providers via OpenAI-compatible endpoints: GPT, Claude, Gemini, Ollama.
- Professional or humorous tone (alternative prompt).

## Architecture (summary)
`cli/app` -> `brochure` (pipeline) -> `scraper` + `providers` (client/model per provider) + `prompts`. See [docs/lld-design.md](docs/lld-design.md).

## Prerequisites
- Python >= 3.11
- At least one of: `OPENAI_API_KEY` (default "GPT"), `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or a local Ollama (`ollama serve`, `ollama pull llama3.2`; no key, no GPU strictly needed).

## Setup
```bash
cp .env.example .env
uv venv && uv pip install -e ".[dev]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

## Run
```bash
brochure-generator "Hugging Face" https://huggingface.co
brochure-generator "Hugging Face" https://huggingface.co --model Claude --tone humorous
python -m brochure_generator "Acme" https://acme.com --model Ollama
brochure-generator --ui        # Gradio app with model dropdown
```

## Models (override via env, see `.env.example`)
| Choice | Brochure model | Link-selection model |
|---|---|---|
| GPT | gpt-4.1-mini | gpt-5-nano |
| Claude | claude-sonnet-4-5-20250929 | claude-haiku-4-5 |
| Gemini | gemini-2.5-flash | gemini-2.5-flash |
| Ollama | llama3.2 | llama3.2 |

Note: the selection step uses the *chosen* provider's fast model so a single key suffices.

## Tests
`pytest` - offline, LLM client and scraper mocked.

## Troubleshooting
- Empty/odd brochure: site is JS-rendered or blocks scrapers (403). Not supported without a headless browser.
- `... is not set`: add the key for the selected provider to `.env`.
- Small Ollama models may return invalid JSON for the link step; the parser tolerates fences/prose but can still fail - use a bigger model.
- The prompt is truncated to 5,000 chars , so later pages may be cut off.
