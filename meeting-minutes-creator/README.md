# meeting-minutes-creator

Upload meeting audio, get structured **markdown minutes** (attendees, summary, key discussion points, takeaways,
action items with owners) with streaming output.

**What you will learn:** Whisper speech recognition plus Llama 3.1 8B in 4-bit on a free Colab T4. Background on Colab,
pipelines, tokenizers, models/quantisation and `visualizer.py` is condensed in [docs/concepts.md](docs/concepts.md).

## Two backends

| | `local` | `api` (fallback, no GPU) |
|---|---|---|
| Speech to text | `openai/whisper-medium` via HF `pipeline` | OpenAI `whisper-1` |
| Minutes | `meta-llama/Meta-Llama-3.1-8B-Instruct`, 4-bit NF4 (bitsandbytes), streamed | OpenAI `gpt-4.1-mini`, streamed |
| Hardware | **CUDA GPU required** (Colab T4 16 GB is enough). bitsandbytes 4-bit does not work on CPU or Apple Silicon | Any machine |
| Keys | `HF_TOKEN` + accepted Llama licence | `OPENAI_API_KEY` |

`BACKEND=auto` (default) picks `local` when torch sees a CUDA GPU, otherwise `api`. You can also paste/supply a
text transcript to skip transcription entirely.

## GPU / Colab requirements

- Run locally on a Linux/Windows box with an NVIDIA GPU (>= ~8 GB VRAM for the 4-bit 8B model plus Whisper
  medium; Colab free T4 works) or in Google Colab: *Runtime -> Change runtime type -> T4 GPU*.
- Colab quick start:
  ```python
  !pip install -q "git+file:///content/meeting-minutes-creator#egg=meeting-minutes-creator[local]"  # or upload the folder and: pip install -e ".[local]"
  from google.colab import userdata; import os; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
  !meeting-minutes run /content/drive/MyDrive/denver_extract.mp3 --backend local
  ```
  (Mounting Google Drive with `drive.mount('/content/drive')` is one way to load audio in Colab.)
- Llama 3.1 is gated: accept the licence on the model page and create a Hugging Face token.
- `ffmpeg` must be on PATH for the HF Whisper pipeline to decode mp3/m4a.
- No GPU? Use `--backend api`.

## Setup

```bash
cd meeting-minutes-creator
uv venv && uv pip install -e ".[dev]"            # API path only (light)
uv pip install -e ".[local]"                     # + torch/transformers/bitsandbytes for the GPU path
uv pip install -e ".[viz]"                       # + token-probability visualiser (optional)
cp .env.example .env
```

## Run

```bash
# no audio handy? the sample transcript exercises the minutes step only
uv run meeting-minutes run --transcript examples/sample_transcript.txt --backend api -o minutes.md

uv run meeting-minutes run my_meeting.mp3 --backend api      # API: transcribe + minutes
uv run meeting-minutes run my_meeting.mp3 --backend local    # GPU: Whisper + Llama 4-bit
uv run meeting-minutes ui                                    # Gradio UI (upload, backend radio, streaming)
python -m meeting_minutes ui
```

No sample audio is included; bring your own mp3/wav. The OpenAI
transcription endpoint accepts files up to 25 MB (compress with `ffmpeg -i in.wav -ac 1 -b:a 48k out.mp3`).

## Optional: how inference works

`src/meeting_minutes/visualizer.py` is a token-prediction graph (OpenAI logprobs + networkx):
```python
from meeting_minutes.visualizer import TokenPredictor, create_token_graph, visualize_predictions
```
Needs the `viz` extra and `OPENAI_API_KEY`.

## Tests

```bash
uv run pytest     # offline; OpenAI clients mocked, no GPU, no torch needed
```
Local-model classes are lazy-loaded and only exercised on a GPU machine.

## Architecture

`transcribe.py` (WhisperLocal / WhisperAPI) -> `pipeline.py` (backend choice, caching) -> `minutes.py`
(LlamaLocal / ChatAPI, streaming) -> `app.py` / `cli.py`. See [docs/lld-design.md](docs/lld-design.md).

## Troubleshooting

- `4-bit bitsandbytes quantisation needs a CUDA GPU`: use `--backend api` or a Colab T4.
- `401/403 ... meta-llama`: accept the Llama licence and set `HF_TOKEN`.
- `CUDA out of memory`: restart runtime; use `openai/whisper-small`; lower `MAX_NEW_TOKENS`; very long transcripts
  need chunking (not implemented).
- Whisper cannot decode the file: install `ffmpeg`.
- `ImportError ... '.[local]'`: install the `local` extra.
