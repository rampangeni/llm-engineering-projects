# Webpage Summarizer

Fetch a web page, strip it down to readable text, and ask an LLM for a short, snarky markdown summary.
The backend is switchable between **OpenAI** and a local **Ollama** model through the same OpenAI-compatible Chat Completions API.

**What you will learn:** the minimal scrape -> prompt -> chat-completion loop, and swapping providers (OpenAI vs local Ollama) through the OpenAI-compatible endpoint.

## Features
- CLI (`webpage-summarizer URL`) and Gradio UI (`--ui`) with streaming output.
- Backend switch: `--backend openai|ollama` or `LLM_BACKEND` env var; models overridable per backend.
- Defensive scraper (timeout, HTTP errors turned into a clear message).
- Offline tests with a mocked LLM client.

## Architecture (summary)
`cli/app` -> `summarizer` (prompts + LLM call) -> `scraper` (requests + BeautifulSoup) and `config` (env -> `Settings` -> OpenAI-SDK client).
Details and diagrams: [docs/lld-design.md](docs/lld-design.md).

## Prerequisites
- Python >= 3.11
- **OpenAI backend:** `OPENAI_API_KEY` (default model `gpt-4.1-mini`).
- **Ollama backend:** [Ollama](https://ollama.com) installed and running (`ollama serve`), and a model pulled: `ollama pull llama3.2` (use `llama3.2:1b` on small machines). No API key or GPU strictly required (CPU works, slowly).

## Setup
```bash
cp .env.example .env        # then edit
uv venv && uv pip install -e ".[dev]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

## Run
```bash
webpage-summarizer https://huggingface.co
webpage-summarizer https://huggingface.co --backend ollama --model llama3.2
python -m webpage_summarizer https://anthropic.com --max-chars 4000
webpage-summarizer --ui          # Gradio app
```

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | - | required for openai |
| `OPENAI_MODEL` | `gpt-4.1-mini` | model for openai |
| `OPENAI_BASE_URL` | unset | any OpenAI-compatible endpoint |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | model for ollama |

## Tests
```bash
pytest          # fully offline, no API key needed
```

## Troubleshooting
- *403 / empty summary*: the site blocks simple scrapers or renders with JavaScript (e.g. openai.com). This scraper does not run JS.
- *Connection refused on :11434*: start Ollama with `ollama serve`.
- *model not found (Ollama)*: `ollama pull <model>`.
- *Gradio opens nothing*: browse to the printed local URL.
