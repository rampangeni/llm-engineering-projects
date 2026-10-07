# FlightAI - Airline AI Assistant

A customer-support chatbot for a fictional airline, "FlightAI". It answers briefly, **calls tools** to look up ticket prices and book tickets (in-memory ticket DB), **generates a pop-art image** of the destination city, and **speaks** its replies. A Gradio UI wires it together; there is also a terminal chat mode and an Ollama backend (text + tools only).

**What you will learn:** a Gradio chatbot with a system prompt, tool/function calling (price lookup and `set_ticket_price`), image generation, text-to-speech, and a multimodal Gradio Blocks UI.

## Features
- System-prompted chat ("short, courteous, 1 sentence, say if you don't know").
- Clean **tool registry**: schema + handler registered together; tools `get_ticket_price`, `book_ticket`, and (opt-in) admin `set_ticket_price`. Parallel tool calls and multi-round loops supported (bounded to 5 rounds).
- City image (`gpt-image-1-mini`) and TTS (`gpt-4o-mini-tts`, voice `onyx`) - degrade silently if the call fails; auto-disabled on Ollama.
- Thread-safe in-memory `TicketStore` (london 799, paris 899, tokyo 1420, sydney 2999). Bookings get ids `FL-0001`...
- Optional basic auth for the UI via env vars (no hardcoded credentials).

## Architecture (summary)
`app/cli` -> `factory` -> `Assistant` (tool loop) -> `ToolRegistry` -> `TicketStore`; `Media` for image/speech. Details and diagrams: [docs/lld-design.md](docs/lld-design.md).

## Prerequisites
- Python >= 3.11.
- `OPENAI_API_KEY` for the default backend (needed for chat, image generation and TTS).
- Optional free backend: Ollama with a tool-capable model (`ollama pull llama3.2`); set `ASSISTANT_BACKEND=ollama`. No image/audio there. A GPU is not required.

## Setup
```bash
cp .env.example .env
uv venv && uv pip install -e ".[dev]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

## Run
```bash
airline-assistant                      # Gradio UI (chat + image + audio)
airline-assistant --chat               # terminal chat
airline-assistant --backend ollama --model llama3.2 --chat
airline-assistant --no-media           # skip image + speech (cheaper)
python -m airline_assistant
```
Try: "How much is a ticket to Tokyo?", "Book me a ticket to Paris, my name is Ada Lovelace", "What's the price to Mars?".
To let the chat set prices (demo of a write tool), set `ENABLE_ADMIN_TOOLS=1` and say "Set the price of Berlin to 499" - do not enable this on a public deployment.

## Tests
`pytest` - offline; the OpenAI client is replaced by a scripted fake.

## Troubleshooting
- `OPENAI_API_KEY is not set`: edit `.env`.
- No image/audio: backend is Ollama, `--no-media` is set, or the API call failed (see console). Image models may require org verification.
- Audio doesn't autoplay: browsers block autoplay until you interact with the page.
- Ollama ignores tools: use a model with tool support (llama3.1/3.2, qwen2.5...).
- Data is lost on restart by design (in-memory).
