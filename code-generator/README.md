# code-generator

Convert Python into high-performance **C++ or Rust** using several frontier and open-source LLMs, then compile and
run the result and compare its runtime against the original Python.

**Concepts covered:**

| Topic | What it covers |
|-----|---------------------|
| C++ generation | Python -> C++ with frontier models (GPT-5, Claude, Grok, Gemini), system probe, compile + run, speedup comparison |
| Open-source models | Ollama / Groq / OpenRouter, Gradio UI with model dropdown |
| Rust and UI | Rust target, richer Gradio UI (`styles.py`), "hard" LCG + max-subarray example, `system_info.py` toolchain detection |

## Features

- 9 models in a dropdown (frontier + Ollama + Groq + OpenRouter) all reached through the `openai` SDK with different `base_url`s.
- **Streaming** generation into the code pane.
- C++ (clang++/g++) and Rust (rustc) targets; compile command picked automatically from a system probe
  (`system_info.py`: OS, CPU/SIMD, compilers, rustup).
- Run Python and run the generated native code from the UI; wall time and speedup are shown.
- Sample programs: Leibniz pi series and the LCG / `max_subarray_sum` "hard" example (`src/code_generator/samples/`).
- CLI: `info`, `port FILE [--run]`, `ui`.

## Architecture (summary)

`models.py` (registry + client factory) -> `prompts.py` -> `generator.py` (streaming call, fence stripping) ->
`runner.py` (temp dir, compile, run, timeout) ; `system_info.py` (probe + compile command) ; `app.py`/`cli.py` front ends.
Details in [docs/lld-design.md](docs/lld-design.md).

## Prerequisites

- Python >= 3.11
- At least one provider key (see `.env.example`) **or** a local Ollama server with a coder model pulled
  (`ollama pull qwen2.5-coder`, `gpt-oss:20b`, ...).
- To run native output: a C++ compiler (`clang++`/`g++`; macOS: `xcode-select --install`) and/or Rust (`https://rustup.rs`).
  Without compilers you can still generate code and paste it into an online compiler.

## Setup

```bash
cd code-generator
uv venv && uv pip install -e ".[dev]"      # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                        # add the keys you have
```

## Run

```bash
uv run code-generator ui --language Rust     # Gradio UI (C++ is the default)
uv run code-generator info                   # what the system probe sees
uv run code-generator port src/code_generator/samples/pi.py --model gpt-5 --run -o main.cpp
python -m code_generator ui
```

Example: pick "lcg + max_subarray_sum (hard)", choose a model, press **Port code**, then **Run Python** and
**Compile & run**. In one run, Python took ~33.8 s vs ~0.0003 s for the best Rust output (results vary by model;
many models got the "hard" example wrong, which is the point - always compare outputs).

## SECURITY: executing generated code

The Run buttons execute Python and compiled native code that an LLM wrote (or that you pasted) **with your user's
privileges**. This project mitigates but does not sandbox: temp working dir, timeouts (`RUN_TIMEOUT`,
`COMPILE_TIMEOUT`), no shell, and API-key-looking env vars are stripped from children. For untrusted input use a
container/VM (e.g. `docker run --rm --network none ...`) or skip running and use an online compiler.

## Tests

```bash
uv run pytest      # offline; LLM clients are mocked. Compile tests skip if no compiler.
```

## Troubleshooting

- `X_API_KEY is not set` in the output pane: add it to `.env` and restart.
- Ollama models fail: check `ollama serve` is up and the model is pulled; `OLLAMA_BASE_URL` overrides the URL.
- `rustc not found`: install rustup and make sure `~/.cargo/bin` is on PATH.
- Timeout: raise `RUN_TIMEOUT`; the Python sample takes ~20-35 s on typical machines.
- `reasoning_effort` errors on non-OpenAI endpoints: it is only sent to the GPT-5 entry.
- Filenames in prompts are `main.rs`/`main.cpp`.
