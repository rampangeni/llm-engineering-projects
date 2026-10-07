# price-predictor-openai-finetune

Part of the **"The Price is Right"** project: fine-tune OpenAI's
`gpt-4.1-nano` to estimate a product's price from its summary, then compare the fine-tuned model with the
base model. Self-contained package `price_openai_ft` with a single CLI, `openai-ft`.

| Step | Here |
|---|---|
| Build jsonl, upload, fine-tune, evaluate | `openai-ft prepare / launch / status / evaluate / compare` |
| Deep neural network training | **not here** - see [`price-dataset-and-baselines`](../price-dataset-and-baselines) (`train-dnn`) |

## Prerequisite: the dataset
Training examples are built from the summarised items dataset produced by
[`price-dataset-and-baselines`](../price-dataset-and-baselines) (`<HF_USERNAME>/items_full` or `items_lite`;
splits `train` / `validation` / `test`, columns `title, category, price, summary, ...`).
By default it reads the pre-built (private) `rampangeni/items_full` (log in with `hf auth login` or set `HF_TOKEN`); set `HF_USERNAME` to use your own.
No dataset is stored in this repo.

## What this project PRODUCES
| Artifact | Format |
|---|---|
| `jsonl/fine_tune_train.jsonl`, `jsonl/fine_tune_validation.jsonl` | OpenAI chat fine-tuning format, one `{"messages":[{"role":"user",...},{"role":"assistant","content":"$12.34"}]}` per line |
| A private OpenAI model `ft:gpt-4.1-nano-2025-04-14:<org>:pricer:<id>` | lives in your OpenAI account; name recorded in `artifacts/ft_state.json` (`job_id`, `train_file_id`, `validation_file_id`, `fine_tuned_model`, ...) |
| optional metrics / predictions JSON | `--output`, `--save-predictions` |

Prompt contract (training and inference are identical except for the assistant turn):
```
user:      Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}
assistant: $<price with 2 decimals>          (training only)
```
Inference uses `max_tokens=7`; the reply is parsed with a number regex.

## Prerequisites
* Python >= 3.11, `uv` or pip. No GPU needed (training runs on OpenAI).
* `OPENAI_API_KEY` with fine-tuning access and billing. `HF_TOKEN` only for private datasets.
* Optional **Weights & Biases**: pass `--wandb-project` and link your W&B API key under *Settings > Integrations* in the OpenAI dashboard (OpenAI pushes the metrics; no `wandb` package needed here).
* Cost: a 20,000-example run cost about $3.42; staying with **100 examples, 1 epoch** costs only cents.

## Setup
```bash
cd price-predictor-openai-finetune
cp .env.example .env
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run
```bash
openai-ft prepare  --train 100 --val 50         # -> jsonl/fine_tune_{train,validation}.jsonl
openai-ft launch   --wandb-project gpt-pricer   # uploads files, creates the job (seed 42, 1 epoch, batch size 1, suffix "pricer")
openai-ft status   --wait                       # follow events; records the fine-tuned model name when done
openai-ft evaluate --model ft   --size 200      # fine-tuned model on the first 200 test items
openai-ft evaluate --model base --size 200      # zero-shot gpt-4.1-nano
openai-ft compare  --size 200 --output compare.json
```
Handy flags (global ones go BEFORE the sub-command, e.g. `openai-ft --lite prepare`): `--lite` (use items_lite), `--dataset user/name`; per-command: `--epochs`, `--batch-size auto`, `--lr-multiplier`, `--no-validation`,
`python -m price_openai_ft ...` is equivalent to `openai-ft ...`.

## Reference results (200 test items from `rampangeni/items_full`, avg absolute error in $)
| Model | Error |
|---|---|
| gpt-4.1-nano zero-shot | 62.51 |
| **gpt-4.1-nano fine-tuned, 100 examples, 1 epoch** | **75.91** (worse than base!) |
| Reference results by training size: mini 200 -> 96.58, mini 2,000 -> 79.29, nano 2,000 -> 82.26, **nano 20,000 -> 67.75** | |
| For context: GPT-5.1 zero-shot 44.74, deep neural network 46.49, QLoRA Llama 3.2 3B (full) 39.85 | |

Take-away: fine-tuning a *frontier* model on a small sample of a task it already does reasonably
well mostly teaches it the output format and does **not** beat the base model - that motivates an open-source
fine-tune (see `price-predictor-qlora-training`). Your numbers will differ (non-deterministic API; only the first 200 test items; fine-tune sampling noise).
The tool also reports MSE, RMSLE, r2 and hit rate (share of green estimates: error < $40 or < 20%).

## Tests
```bash
pytest     # offline: jsonl building/validation, job kwargs, polling with a fake client, metrics, CLI wiring
```

## Troubleshooting
* *Job fails at validation*: OpenAI needs >= 10 examples; `prepare` refuses fewer. Re-run `validate_jsonl` (done automatically on launch).
* *`model not found` on `--model ft`*: the job has not succeeded yet - run `openai-ft status --wait`.
* *`gpt-4.1-nano-2025-04-14` not available to fine-tune*: your org may lack access; pass `--model <snapshot>` for any supported chat model.
* *Reply not numeric*: raise `--max-tokens`; unparseable replies score as 0.
* W&B integration silently absent: link the key in the OpenAI dashboard first.

## Verified / not verified
Verified: unit tests (21), `py_compile`, CLI `--help`. **Not verified**: real uploads, a real fine-tuning job, or reproducing the
reference error numbers (needs paid OpenAI access).

Design: [docs/lld-design.md](docs/lld-design.md).
