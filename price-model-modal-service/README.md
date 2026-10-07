# price-model-modal-service

Serverless GPU deployment of the fine-tuned **Llama 3.2 3B QLoRA price predictor** on [Modal](https://modal.com),
with a small Python client and CLI to call it. This is the "serving" half of the *The Price Is Right* capstone:
it takes a short product description and returns an estimated price in dollars.

Downstream consumer: `../price-is-right-deal-hunter` (calls the deployed app by name).

## Prerequisites: how the models are trained

This project **serves** a model; it does not train one. The model it serves comes from:

| Training project | What it produces | What this service consumes |
|---|---|---|
| [`../price-predictor-qlora-training`](../price-predictor-qlora-training) | A LoRA adapter pushed to the Hugging Face Hub as `{HF_USER}/{PROJECT_NAME}-{RUN_NAME}` (default naming `price-<timestamp>`), with a specific commit **revision** | `HF_USER`, `PROJECT_NAME`, `RUN_NAME` (or `ADAPTER_REPO`) and `ADAPTER_REVISION` (the exact commit hash of the adapter repo) |
| [`../price-dataset-and-baselines`](../price-dataset-and-baselines) | The items dataset and baselines (not used directly here) | nothing |

The service loads `BASE_MODEL` (`meta-llama/Llama-3.2-3B`, a **gated** repo: accept Meta's licence on the HF model page
with the account whose token you give Modal) in 4-bit NF4, then applies the adapter at `ADAPTER_REVISION`.
The prompt format (`What does this cost to the nearest dollar?\n\n<text>\n\nPrice is $`) is the one used in training;
see `src/price_model_service/prompts.py`.

**Fallbacks so you can run without your own training run**

1. *Default:* the `.env.example` defaults point at a pre-built (private) adapter; the HF token in your Modal secret must be able to read it
   (`rampangeni/price-2025-11-28_18.47.07` @ `3e6de25...`), so deployment works with no training at all.
2. `BASE_ONLY=1`: serve the plain base model with no adapter (proves the GPU/secret/pipeline; prices will be poor).
3. `PRICER_BACKEND=mock` (client side): a deterministic offline stub, no Modal, no GPU, no network.

## Features

- Step-by-step teaching apps: `hello` (CPU, region pinning) -> `llama_smoke` (T4 + HF secret) ->
  `pricer_ephemeral` (adapter, `app.run()`), then the **deployed** apps.
- `service.py` (class `Pricer`): model loaded once in `@modal.enter()`, HF cache on a **Modal Volume**,
  `min_containers` (keep warm) and `scaledown_window` (idle shutdown) driven by env vars.
- `service_function.py`: the simpler function variant that reloads the model per call (educational; slow).
- `PricerClient`: looks up the deployed class/function **by name**, retries with backoff, optional LLM rewrite step.
- `Preprocessor` (litellm): rewrites a messy blurb into Title/Category/Brand/Description/Details; default
  `ollama/llama3.2`, or e.g. `groq/openai/gpt-oss-20b` via `PRICER_PREPROCESSOR_MODEL`.
- `price-model` CLI: `config`, `deploy`, `price`, `preprocess`, `warm`, `cool`, `run-ephemeral`.

## Architecture (summary)

```
 your code / CLI ──► PricerClient ──(modal.Cls.from_name "pricer-service","Pricer")──► Modal container (T4)
                         │                                                     ├─ Volume hf-hub-cache (/cache)
                         └─ Preprocessor (litellm, optional)                   ├─ Secret huggingface-secret (HF_TOKEN)
                                                                               └─ 4-bit Llama-3.2-3B + LoRA adapter
```
Full detail, diagrams and design rationale: [docs/lld-design.md](docs/lld-design.md).

## Requirements and credentials

| Need | Used for | How |
|---|---|---|
| Python >= 3.11 | | |
| Modal account + token | deploy and call | `modal token set --token-id ak-... --token-secret as-...` (or `modal token new`) |
| Hugging Face token | download gated Llama + adapter | Modal dashboard -> Secrets -> *Hugging Face* template, **named `huggingface-secret`**, key `HF_TOKEN` (change the name with `HF_SECRET_NAME`) |
| Ollama (default) **or** a hosted-LLM key (`OPENAI_API_KEY`, `GROQ_API_KEY`...) | optional preprocessor only | `PRICER_PREPROCESSOR_MODEL` |

## Setup

```bash
cd price-model-modal-service
uv venv && uv pip install -e ".[dev]"        # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                         # edit adapter / names if you trained your own
uv run modal token set --token-id ak-... --token-secret as-...
```

## Run

```bash
# 1. Ladder of ephemeral demos (each spins an app up and tears it down)
uv run price-model run-ephemeral hello
uv run price-model run-ephemeral llama "Hey Jude, don't make it"
uv run price-model run-ephemeral pricer "Quadcast HyperX condenser mic, connects via usb-c"

# 2. Deploy (first build can take ~10 min; later cold starts ~30 s, warm calls ~2 s)
uv run price-model deploy                    # == modal deploy -m price_model_service.service
uv run price-model deploy --variant function # slow, reload-per-call variant (same app name; replaces the class app)

# 3. Call it
uv run price-model price "Shure MV7+ professional podcaster microphone with usb-c and XLR outputs"
uv run price-model price --preprocess "iPhone 10"          # rewrite first (needs Ollama or API key)
uv run price-model price --backend mock "anything"         # offline

# 4. Latency / cost control
uv run price-model warm --minutes 20         # stay warm 20 min after the last call (default 2 min)
uv run price-model cool                      # revert
#   or set MIN_CONTAINERS=1 in .env and redeploy to keep one container always on (bills continuously!)
```

Python usage:

```python
from price_model_service.client import PricerClient
print(PricerClient().price("Samsung 55 inch 4K UHD smart TV"))
```

Other projects can skip this package entirely: `modal.Cls.from_name("pricer-service", "Pricer")().price.remote(text)`.

## Tests

```bash
uv run pytest        # offline: Modal, litellm and the network are mocked / never contacted
```
Covers: prompt format and price parsing, settings (incl. base-only fallback), client retry/backoff and name lookup,
autoscaler calls, preprocessor usage tracking, importability of the Modal app definitions, and the CLI.

## Troubleshooting

- **`Secret 'huggingface-secret' not found`**: create it in the Modal dashboard (or set `HF_SECRET_NAME`, e.g. `hf-secret`).
- **401/403 on `meta-llama/Llama-3.2-3B`**: request access on the model page using the HF account of the token.
- **`App 'pricer-service' not found` from the client**: deploy first, and make sure `MODAL_APP_NAME` is identical on both sides.
- **Class vs function mismatch**: `modal-class` expects the `service` app; `modal-function` expects `service_function`.
- **Slow first call**: image build + weight download (cached on the Volume afterwards) + 4-bit load; use `warm`.
- **Unexpected GPU bill**: `MIN_CONTAINERS` > 0 or a long `warm` keeps a GPU running; `modal app stop pricer-service` removes it.
- **`ModuleNotFoundError: price_model_service` inside the container**: install the package (`pip install -e .`) before `deploy`; the source is shipped via `add_local_python_source`.

## Verification status

Verified offline: unit tests, `compileall`, importing/defining the Modal apps. **Not verified** (needs a Modal account,
GPU and HF access): image build, actual deployment, remote inference, Volume caching, `update_autoscaler`.
