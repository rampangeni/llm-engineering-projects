# price-dataset-and-baselines

Part 1 of the **"The Price is Right"** project: build the dataset of Amazon products with prices, measure how hard the
problem is with baseline predictors, and **train the deep neural network that produces
`deep_neural_network.pth`**.

This is a self-contained package (`price_dataset`) with CLIs - no notebooks, no external repo imports.

| Step | Here |
|---|---|
| Data curation | `curate-items` (`price_dataset.parser/loaders/curate`) |
| LLM pre-processing (summaries) | `summarize-items` (`price_dataset.batch/preprocessor`) |
| Evaluation, baselines, traditional ML | `run-baselines` (`price_dataset.baselines/evaluator`) |
| Human, neural net, frontier LLMs | `human-baseline`, `eval-frontier`, `price_dataset.dnn` |
| DNN training and evaluation | `train-dnn`, `eval-dnn` |
| Results | benchmark table below |

Next in the series: [`price-predictor-openai-finetune`](../price-predictor-openai-finetune) and
[`price-predictor-qlora-training`](../price-predictor-qlora-training). Both load the dataset produced here.

## What this project PRODUCES (contract for downstream projects)

| Artifact | Where | Format |
|---|---|---|
| `<HF_USERNAME>/items_raw_full`, `items_raw_lite` | HF Hub dataset | `DatasetDict` train/validation/test (800,000 / 10,000 / 10,000 and 20,000 / 1,000 / 1,000). Columns: `title, category, price, full, weight, summary(None), prompt(None), id(None)` |
| `<HF_USERNAME>/items_full`, `items_lite` | HF Hub dataset | Same splits/columns after summarisation: `full` and `id` are `None`, **`summary` is filled** (`Title:/Category:/Brand:/Description:/Details:` block). **This is the dataset every other project trains/evaluates on.** |
| `deep_neural_network.pth` | `artifacts/` (or `--out`), optionally a HF model repo | `torch.save(model.state_dict())` of `price_dataset.dnn.model.DeepNeuralNetwork(input_size=5000, num_layers=10, hidden_size=4096, dropout_prob=0.2)` (about 289M params, 1.1 GB). Input = 5000-dim binary `HashingVectorizer(n_features=5000, stop_words="english", binary=True)` of `item.summary`. Output = normalised log-price. |
| `deep_neural_network.meta.json` | next to the `.pth` | `y_mean`, `y_std`, architecture, vectorizer config, training history, test metrics. |

**Decoding the DNN output** (needed by any consumer, e.g. a serving service):
`price = max(0, exp(output * y_std + y_mean) - 1)`. For the reference full-dataset weights
`y_mean = 4.434937953948975`, `y_std = 1.0328539609909058` (typically hard-coded as `Y_MEAN`/`Y_STD` by consumers).
The `.pth` alone does **not** contain them - that is why the sidecar JSON exists. Inference helper:
`price_dataset.dnn.runner.DeepNeuralNetworkPredictor(weights_path)` (falls back to the default constants when the sidecar is missing).

Hub namespace is controlled by `HF_USERNAME` (default `rampangeni`, which holds
the pre-built `items_*` datasets - you can skip curation + summarisation entirely and just train/evaluate on them).

## Prerequisites

* Python >= 3.11, [`uv`](https://docs.astral.sh/uv/) (or pip).
* `HF_TOKEN` (or `hf auth login`): write access to push datasets/weights; read access is also needed to load `rampangeni/items_*` because those repos are private.
* `GROQ_API_KEY` for batch summarisation (`summarize-items`); typical cost: < $1 lite, ~ $30 full.
* Frontier comparison: the API key of each provider you test (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`).
* DNN training: an accelerator is strongly recommended. The reference run trained 5 epochs on 800k items in ~4 h on an M1 Mac (MPS). A CUDA GPU (Colab/Modal A100/T4) works too; CPU only for smoke tests. RAM: the sparse pipeline here needs ~ 2-3 GB for data plus ~ 3.5 GB for weights + AdamW state at hidden size 4096.
* Curation of the full dataset downloads many GB from `McAuley-Lab/Amazon-Reviews-2023` (Electronics etc.) and takes hours. Requires `datasets<4` because that dataset uses a loading script (`trust_remote_code`).

## Setup

```bash
cd price-dataset-and-baselines
cp .env.example .env            # fill in what you need
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,baselines,dnn,frontier,summarize,plots]"   # or pick extras
# pip equivalent: pip install -e ".[all,dev]"
```

Extras: `summarize` (litellm, groq), `baselines` (scikit-learn, pandas, xgboost), `dnn` (torch, scikit-learn, scipy),
`frontier` (litellm), `plots` (plotly). macOS + XGBoost may need `brew install libomp`.

## Run

All commands accept `--lite` (20k-item dataset, free/fast) or `--dataset user/name`.

```bash
# Curate (hours for the full set; --push uploads to <HF_USERNAME>/items_raw_{full,lite})
curate-items --categories Appliances --sample-size 2000        # tiny dry run
curate-items --push                                              # the real 820k-item run

# Summarise with an LLM (Groq Batch API) and publish items_{full,lite}
summarize-items submit --lite
summarize-items fetch  --lite --wait
summarize-items push   --lite                                    # needs HF_USERNAME = your account
summarize-items direct --lite --limit 20 --model ollama/llama3.2   # no batch API, any litellm provider

# Baselines (uses rampangeni/items_full unless you point elsewhere)
run-baselines --lite --size 200
run-baselines --only random constant linear bow xgb --output results.json --plot-dir work/plots

# Human + frontier models
human-baseline export --n 100 ; human-baseline eval --csv human_out.csv
human-baseline eval --csv data/human_guesses.csv --dataset rampangeni/items_full   # a sample of 100 human price guesses
eval-frontier --model openai/gpt-4.1-nano --size 200
eval-frontier --model gemini/gemini-3-pro-preview --size 50 --workers 2

# Train the DNN (the artifact downstream consumers use)
train-dnn --lite --epochs 2 --hidden-size 512 --out artifacts/dnn_smoke.pth      # CPU smoke test
train-dnn --epochs 5 --out artifacts/deep_neural_network.pth                      # reference run
train-dnn --epochs 5 --push-to-hub <you>/price-dnn                                # optional upload
eval-dnn  --weights artifacts/deep_neural_network.pth --size 200
```

## Reference benchmarks (average absolute error in $ on 200 test items, `rampangeni/items_full`)

Reference numbers for comparison:

| Model | Avg error |
|---|---|
| Constant (training mean) | 106.18 |
| Linear regression (weight, text length) | 101.56 |
| Bag-of-words + linear regression | 76.81 |
| Random forest (15k subset) | 72.28 |
| XGBoost | 68.23 |
| Human | 87.62 |
| 8-layer vanilla neural network | 63.97 |
| GPT-4.1 nano (zero-shot) | 62.51 |
| Grok 4.1 Fast | 57.62 |
| Gemini 3 Pro | 50.54 |
| Claude 4.5 (the code calls `claude-opus-4-5`) | 47.10 |
| GPT-5.1 | 44.74 |
| GPT-4.1 nano fine-tuned (100 examples, see openai-finetune project) | 75.91 |
| **Deep neural network (this project)** | **46.49** |

Other reference facts: random forest on all 800k items took ~15 h and reached $56.40. DNN training log
(800k train, 1,000 val, MPS, 289,128,449 parameters): val MAE per epoch
$58.26, $56.25, $56.55, $55.70, $53.15; learning rate 0.001 -> 0.000655 after 5 epochs. Random/constant baselines
were evaluated with 200 samples; the random guesser is intentionally terrible (no number stored).
Absolute numbers will differ slightly for your own run (seeds fixed to 42, but hardware/library differences apply).

Metrics reported by `price_dataset.evaluator`: average absolute error, MSE, **RMSLE**, r2 (%), and **hit rate**
(share of "green" estimates: error < $40 or < 20%; orange: < $80 or < 40%; red otherwise).

## Tests

```bash
pytest            # offline: parsing, prompts, sampling/splits, evaluator metrics, batch logic (fake Groq client), DNN target maths
```
`tests/test_dnn_torch.py` trains a tiny network on CPU if torch + scikit-learn are installed (skipped otherwise).

## Design notes

* `Batch` (Groq): injected client, JSON state instead of pickle, fixed `open('rb', encoding=...)` bugs, id->item dict.
* DNN keeps the bag-of-words matrix sparse and densifies per batch (a fully dense tensor would be ~16 GB).
* DNN saves `deep_neural_network.meta.json` (y_mean/y_std) alongside the weights.
* Evaluator metrics are pure Python and add RMSLE + hit rate; charts are written to HTML files, not shown inline.
* `get_features` treats `weight=None` as 0.

## Troubleshooting

* `trust_remote_code is no longer supported` -> install `datasets<4` (`uv pip install "datasets==3.6.0"`).
* `ImportError: libomp` with xgboost on macOS -> `brew install libomp`, or skip `xgb` via `--only`.
* DNN OOM on GPU -> lower `--hidden-size` or `--batch-size` (weights then differ in shape; record this in the meta file, which is done automatically).
* Loading a `.pth` on a CPU-only machine: `eval-dnn --weights deep_neural_network.pth --device cpu` (meta falls back to the default constants).
* Groq batches stay `in_progress` for hours: just re-run `summarize-items fetch --wait`.

## Verified / not verified

Verified: unit tests, `py_compile`, every CLI `--help`, a tiny CPU DNN train/save/load round-trip.
**Not verified** (needs network, keys, hours of compute): downloading Amazon data, Groq batches, pushing to the Hub,
real frontier calls, full-scale DNN training and the reference benchmark numbers.

Design details: [docs/lld-design.md](docs/lld-design.md).
