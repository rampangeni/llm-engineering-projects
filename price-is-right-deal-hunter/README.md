# price-is-right-deal-hunter

An autonomous, multi-agent **deal hunter**. It scans deal RSS feeds, asks an LLM to pick the 5 best-described deals,
estimates each product's *true* price with an **ensemble** (a fine-tuned Llama on Modal, a RAG pipeline with a frontier
model over a Chroma vector store, and a deep neural network), and sends a push notification (Pushover) when a deal is
priced far below its estimate. A Gradio app shows found deals, live colour-coded agent logs and a 3-D t-SNE plot of the
vector store, and re-runs on a timer.

## Prerequisites: how the models are trained

The agents here **use** models that are **trained in earlier projects** (dataset, baselines and fine-tuning). Run those first, or use the
fallbacks below.

| Consumed artifact | Produced by | How it is referenced here | Fallback if you have not run it |
|---|---|---|---|
| **Items dataset** on the HF Hub (`train` split with `summary`, `category`, `price`) | [`../price-dataset-and-baselines`](../price-dataset-and-baselines) | `HF_ITEMS_DATASET` (default `rampangeni/items_full`, or `items_lite`) -> `deal-hunter build-vectorstore` fills Chroma, which the **FrontierAgent** retrieves from | Use the public pre-built dataset (the default), `--lite --limit 5000` for a quick start, or `ENABLE_FRONTIER=0` |
| **`deep_neural_network.pth`** (state dict of the deep residual MLP) | [`../price-dataset-and-baselines`](../price-dataset-and-baselines) | `NN_WEIGHTS_PATH` (default `./deep_neural_network.pth`) -> **NeuralNetworkAgent** | `ENABLE_NN=0`, or just leave the file out: the ensemble logs a warning and drops that component (weights renormalise) |
| **Fine-tuned LoRA adapter** of Llama 3.2 3B on the HF Hub, pinned to a commit **revision** | [`../price-predictor-qlora-training`](../price-predictor-qlora-training) (the OpenAI-fine-tune variant, [`../price-predictor-openai-finetune`](../price-predictor-openai-finetune), is a comparison, not used here) | Deployed as a Modal app by [`../price-model-modal-service`](../price-model-modal-service) (`ADAPTER_REPO`/`HF_USER`+`RUN_NAME`, `ADAPTER_REVISION`). This project only needs the **names**: `MODAL_APP_NAME=pricer-service`, `MODAL_CLASS_NAME=Pricer` -> **SpecialistAgent** | `SPECIALIST_BACKEND=mock` (deterministic offline stub), `ENABLE_SPECIALIST=0`, or deploy the service with the public pre-built adapter (its default) |

Minimum viable setups: **zero models** `deal-hunter ui --demo` (canned deals, mock pricer, no keys);
**no training of your own** (pre-built dataset + adapter, plus pre-trained NN weights);
**full** (everything trained and deployed by the sibling projects).

## Features

- **ScannerAgent**: RSS + page scraping (dealnews), OpenAI **Structured Outputs** (`DealSelection`) -> 5 best deals; skips URLs already in memory.
- **SpecialistAgent**: calls the deployed Modal function **by app/class name** (no import of the service package); `mock` backend.
- **FrontierAgent**: RAG - SentenceTransformer `all-MiniLM-L6-v2` -> top-5 similar priced products from Chroma -> GPT price.
- **NeuralNetworkAgent**: local deep residual MLP on hashed bag-of-words.
- **EnsembleAgent**: optional LLM rewrite (litellm) -> weighted combination (default 0.8 frontier / 0.1 specialist / 0.1 NN, configurable) with graceful degradation and an optional `min` strategy.
- **MessagingAgent**: Pushover pushes; Claude (via litellm) crafts the text; dry-run when credentials are absent.
- **PlanningAgent** (fixed workflow) and **AutonomousPlanningAgent** (LLM tool-calling loop with `scan` / `estimate` / `notify` tools).
- **DealAgentFramework**: memory (`memory.json`), Chroma, planner wiring. **Gradio UI** with live logs, 3-D t-SNE plot, auto-refresh timer, click a row to re-send an alert.
- `build-vectorstore`, run any agent on its own, `check` (prerequisite report).

## Architecture (summary)

```
RSS feeds -> ScannerAgent -(5 Deals)-> PlanningAgent / AutonomousPlanningAgent
                                         |  for each deal
                                         v
              EnsembleAgent: Preprocessor -> Frontier(RAG: Chroma + GPT) 0.8
                                          -> Specialist(Modal Llama QLoRA) 0.1
                                          -> NeuralNetwork(.pth)         0.1
                                         |  estimate - price = discount
                                         v  > DEAL_THRESHOLD
                                   MessagingAgent -> Pushover      memory.json <- Opportunity
```
Diagrams, tool-calling loop and rationale: [docs/lld-design.md](docs/lld-design.md).

## Requirements

Python >= 3.11. Install only what you use (extras): `rag` (chromadb, sentence-transformers), `nn` (torch, scikit-learn),
`modal`, `ui` (gradio, plotly, scikit-learn), `build` (datasets), `all`.

| Credential | Needed for |
|---|---|
| `OPENAI_API_KEY` | scanner (`gpt-5-mini`), frontier + autonomous planner (`gpt-5.1`) |
| `ANTHROPIC_API_KEY` (or change `MESSAGING_MODEL`) | crafted notification text (`notify`, autonomous planner) |
| Modal token (`modal token set ...`) + deployed `pricer-service` | specialist |
| `PUSHOVER_USER`, `PUSHOVER_TOKEN` | real pushes (otherwise dry-run) |
| `HF_TOKEN` | `build-vectorstore` for private datasets |
| Ollama running (`llama3.2`) **or** `PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b` + `GROQ_API_KEY` | preprocessor (`ENABLE_PREPROCESSOR=0` to skip) |

## Setup

```bash
cd price-is-right-deal-hunter
uv venv && uv pip install -e ".[all,dev]"     # or pick extras; pip works too
cp .env.example .env                           # fill in keys
uv run deal-hunter check                       # shows what is available / missing
```

## Run

```bash
# 0. Try everything offline (canned deals, mock pricer, dry-run notifications)
uv run deal-hunter ui --demo
uv run deal-hunter run-once --demo

# 1. Vector store for the FrontierAgent (30+ min for the full 800k on a GPU; use the lite set to start)
uv run deal-hunter build-vectorstore --lite --limit 20000
uv run deal-hunter build-vectorstore --rebuild                    # full HF_ITEMS_DATASET

# 2. Other prerequisites: put deep_neural_network.pth next to where you run, deploy ../price-model-modal-service

# 3. Individual agents
uv run deal-hunter agent scan
uv run deal-hunter agent specialist "Quadcast HyperX condenser mic, usb-c"
uv run deal-hunter agent frontier   "Shure MV7+ professional podcaster microphone"
uv run deal-hunter agent nn         "Shure MV7+ professional podcaster microphone"
uv run deal-hunter agent ensemble   "Shure MV7+ professional podcaster microphone"
uv run deal-hunter agent message    "Samsung 60 inch LED TV"          # dry-run without Pushover
uv run deal-hunter agent plan                                          # fixed workflow
uv run deal-hunter agent autonomous                                    # LLM tool-calling loop

# 4. The whole app
uv run deal-hunter run-once                  # one planner pass (PLANNER=workflow|autonomous)
uv run deal-hunter ui                        # Gradio; runs on load and every UI_REFRESH_SECONDS
uv run deal-hunter reset-memory              # back to 2 example deals
```

State files in the working directory: `memory.json` (surfaced deals; a 2-deal seed is bundled and used by `reset-memory`),
`products_vectorstore/` (Chroma), `deep_neural_network.pth` (not committed).

## Tests

```bash
uv run pytest      # offline; OpenAI, Modal, litellm, Pushover, RSS and the network are faked
```
Covers deal parsing/scraping, scanner (structured output, memory filtering), ensemble weighting/degradation, frontier RAG
prompt, specialist by-name Modal lookup, messaging, workflow planner, autonomous **tool loop** (scripted tool calls, duplicate
notify, bad args, step cap), memory, vector-store build (fake and real Chroma if installed), log colour mapping, UI streaming
helpers, CLI. Torch/chromadb/gradio tests are skipped automatically if those extras are not installed.

## Troubleshooting

- **`NN weights ... not found`**: obtain `deep_neural_network.pth` from `../price-dataset-and-baselines` or set `ENABLE_NN=0`.
- **Frontier returns nonsense / `Collection is empty`**: run `build-vectorstore`; the UI plot says "Vector store is empty".
- **`App 'pricer-service' not found`**: deploy `../price-model-modal-service` (names must match) or `SPECIALIST_BACKEND=mock`.
- **`reasoning_effort` errors**: your `FRONTIER_MODEL`/`SCANNER_MODEL` does not support it; set `FRONTIER_REASONING_EFFORT=` (empty) and adjust models.
- **No push arrives**: check `deal-hunter check`; without both Pushover variables it logs "Dry run".
- **Preprocessor connection refused**: start Ollama (`ollama pull llama3.2`) or set `ENABLE_PREPROCESSOR=0`.
- **No deals found**: all feed URLs are already in `memory.json` (`reset-memory`) or the site layout changed (`scraping.py`).
- **Many Gradio runs overlap**: the timer fires every 5 minutes; raise `UI_REFRESH_SECONDS` or wait for a run to finish.

## Verification status

Verified here: offline unit/integration tests (incl. real Chroma with a fake encoder, Gradio app construction, torch DNN
forward/load round-trip), `compileall`, the `--demo` pipeline. **Not verified** (needs keys, GPU, network or deployment):
live OpenAI/Anthropic/Groq/Ollama calls, the Modal call, real RSS scraping against dealnews, Pushover delivery,
SentenceTransformer download/embedding, the HF dataset download, the real `.pth` weights, and an interactive Gradio session.
