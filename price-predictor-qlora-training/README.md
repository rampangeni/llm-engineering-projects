# price-predictor-qlora-training

Part of the **"The Price is Right"** project: fine-tune an
open-source LLM (**Llama 3.2 3B**, 4-bit **QLoRA**) to predict a product's price from its summary, track it with
Weights & Biases, push LoRA checkpoints to the HuggingFace Hub, then evaluate and run inference with the adapter.
Self-contained package `price_qlora` with CLIs `qlora-prepare-prompts`, `qlora-quant-compare`, `qlora-train`,
`qlora-evaluate`, `qlora-infer`.

| Step | Here |
|---|---|
| QLoRA, quantisation, base model | `qlora-quant-compare`, `qlora-evaluate --base-only`, [docs](docs/lld-design.md#7-qlora-training-pipeline) |
| Prompt data + base model | `qlora-prepare-prompts` (token cut-off 110, prompt/completion dataset) |
| Training | `qlora-train` |
| Evaluation and inference | `qlora-evaluate`, `qlora-infer` |
| Metrics and item helpers | `price_qlora.evaluator`, `price_qlora.items` |

> **Note on hyper-parameters.** The pipeline targets base `meta-llama/Llama-3.2-3B` with 4-bit NF4 quantisation,
> `PeftModel.from_pretrained(..., revision=...)` for loading adapters and `max_new_tokens=5` at inference.
> Hyper-parameters are standard QLoRA defaults (LoRA r=32 / alpha=64 / dropout 0.1 on q,k,v,o;
> lr 1e-4 cosine, 1 epoch...); tune them for your data.

## Prerequisite: datasets
Needs the summarised items dataset from [`price-dataset-and-baselines`](../price-dataset-and-baselines)
(`<user>/items_full` / `items_lite`; splits train/validation/test with a `summary` column). From it,
`qlora-prepare-prompts` builds the *prompt dataset* (`items_prompts_{full,lite}`) that training reads. Pre-built (private; log in with `hf auth login` or set `HF_TOKEN` to read them)
ones, `rampangeni/items_prompts_full|lite`, are used by default (`DATASET_USERNAME`), so you can skip straight to training.

## What this project PRODUCES (contract for downstream projects)
| Artifact | Format |
|---|---|
| `<HF_USERNAME>/items_prompts_{full,lite}` (optional, from `qlora-prepare-prompts`) | HF dataset, splits **`train` / `val` / `test`**, columns `prompt`, `completion`. `prompt` = `What does this cost to the nearest dollar?\n\n{summary truncated to 110 tokens}\n\nPrice is $`; `completion` = `"<rounded>.00"` for train/val, the exact price string (e.g. `"49.99"`) for test. |
| **`<HF_USERNAME>/price-<RUN_NAME>`** (HF *model* repo, private by default), `RUN_NAME` = `YYYY-MM-DD_HH.MM.SS` | A **PEFT LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`) for the base model `meta-llama/Llama-3.2-3B`; one **git revision (commit hash) per saved checkpoint** (`hub_strategy="every_save"`). Load with `PeftModel.from_pretrained(base_4bit, "<user>/price-<RUN_NAME>", revision="<commit>")`. A serving service should pin the same adapter repo and revision (e.g. as `FINETUNED_MODEL` / `REVISION`). |
| `artifacts/price-<RUN_NAME>/final/`, `run.json` | local copy of the final adapter and the resolved settings + latest hub revision |
| W&B project `price` (optional) | loss curves per run |

Inference contract (shared with downstream serving): prompt `"{QUESTION}\n\n{summary}\n\nPrice is $"`, greedy decode `max_new_tokens=5`, parse the first number of the continuation.

## Prerequisites
* **GPU with CUDA** for training/evaluation (bitsandbytes does not run on Mac/CPU). Memory: Llama 3.2 3B in 4-bit takes roughly 2-3 GB for weights; with LoRA + activations, batch 4 (grad-accum 4) fits a free **Colab T4 (16 GB)**, batch 16 an **A100**. The lite dataset (20k rows, ~1,250 steps at batch 16) is the practical Colab run; the full 800k-row epoch (50,000 steps at batch 16) needs an A100 for hours (the "Fine-tuned Full" run) - see cost notes in the LLD. These sizes/times are estimates, not measured here.
* `HF_TOKEN` with **write** access, and access granted to the gated `meta-llama/Llama-3.2-3B` repo (accept the licence on its page). `HF_USERNAME` = where adapters are pushed.
* `WANDB_API_KEY` (optional; `--no-wandb` to disable).
* Python >= 3.11. Verified API combination: `transformers 4.57`, `trl 0.29`, `peft 0.19` (pyproject pins `transformers<5`, `trl<1`).

## Setup
Local GPU box / Modal / any Linux CUDA machine:
```bash
cd price-predictor-qlora-training
cp .env.example .env              # HF_TOKEN, HF_USERNAME, WANDB_API_KEY ...
uv venv && source .venv/bin/activate
uv pip install -e ".[gpu,dev]"
```
CPU-only (data prep, tests, `--dry-run`): `uv pip install -e ".[dev]"` and add `transformers` if you want `qlora-prepare-prompts`.

### Colab / GPU path
1. Runtime > Change runtime type > GPU (T4 free; A100 on Colab Pro).
2. Upload this folder (or `git clone` your repo) and in a cell:
```python
!pip install -q -e "/content/price-predictor-qlora-training[gpu]"
import os
from google.colab import userdata
for k in ("HF_TOKEN", "WANDB_API_KEY"): os.environ[k] = userdata.get(k)   # Colab secrets
os.environ["HF_USERNAME"] = "your-hf-username"
!qlora-train --lite --batch-size 4 --grad-accum 4 --epochs 1          # smoke run on a T4
!qlora-evaluate --adapter your-hf-username/price-<RUN> --lite --size 100
```
3. Colab disconnects after idle time: keep `--save-steps` small enough (default 2000) - every save is pushed to the Hub, and you can resume with `--resume-from-checkpoint`.
Modal / RunPod: same commands inside a container with a CUDA image (`pip install -e .[gpu]`).

## Run
```bash
# (optional) rebuild the prompt dataset from your own summarised items
qlora-prepare-prompts --lite --dry-run             # token stats: how many summaries exceed 110 tokens
qlora-prepare-prompts --lite --push                # -> <HF_USERNAME>/items_prompts_lite
# What does quantisation cost? (memory per precision, optional baseline error)
qlora-quant-compare --presets 4bit 8bit bf16
qlora-evaluate --base-only --size 200               # untuned 4-bit Llama (reference: $110.72)
# Train
qlora-train --dry-run                               # prints the resolved config (no GPU needed)
qlora-train --lite                                  # 20k rows
qlora-train                                         # full 800k rows (A100)
# Evaluate and use the adapter
qlora-evaluate --adapter <HF_USERNAME>/price-<RUN_NAME> --revision <commit> --size 200 --output metrics.json
qlora-infer --adapter <HF_USERNAME>/price-<RUN_NAME> --revision <commit> --text "Title: ... Details: ..."
qlora-infer --adapter <...> --test-index 3
```
Every command is also runnable as `python -m price_qlora.<train|evaluate|infer|prepare_prompts|quant_compare>`.

**Choosing the revision**: open the model repo's *Commits* tab (or `HfApi().list_repo_commits(repo)`); pick the checkpoint with the lowest eval loss / best `qlora-evaluate` error; pin that commit hash everywhere downstream.

## Default training configuration (`TrainSettings`)
LoRA r=32, alpha=64, dropout 0.1, bias none, targets `q_proj k_proj v_proj o_proj` (about 18.4M trainable parameters on a 3B Llama, ~0.6%);
4-bit NF4 + double quantisation, compute dtype bf16 if the GPU supports it else fp16; 1 epoch, per-device batch 16, lr 1e-4 cosine, warm-up 3%, weight decay 0.01,
grad-norm 0.3, `paged_adamw_32bit`, `max_length` 160, completion-only loss, eval loss on the first 1,000 validation rows every 2,000 steps, save+push every 2,000 steps, seed 42.
Override with flags (`qlora-train --help`) or switch base model with `BASE_MODEL=meta-llama/Llama-3.1-8B` (about 27M LoRA params; needs a larger GPU, and a different tokenizer means the prompt cut-off statistics change).

## Reference results (avg absolute error in $ over 200 test items)
| Model | Error |
|---|---|
| Untuned Llama 3.2 3B, 4-bit | 110.72 |
| Fine-tuned on **lite** (20k) | 65.40 |
| Fine-tuned on **full** (800k) | **39.85** |
| For comparison: best frontier zero-shot (GPT-5.1) 44.74; deep neural net 46.49; fine-tuned GPT-4.1 nano 75.91 | |

The fine-tuned 3B model beats every frontier model tried zero-shot. Your numbers will vary with data size, seed and checkpoint.
`qlora-evaluate` also prints MSE, RMSLE, r2 and hit rate (share of green estimates: error < $40 or < 20%).

## Tests
```bash
pytest     # offline, no GPU / network: prompt+completion formatting and truncation (fake tokenizer), number parsing,
           # metrics, config -> SFTConfig/LoraConfig kwargs, run/hub naming, quant presets, CLI wiring, --dry-run
```

## Troubleshooting
* `401/403 ... meta-llama` -> accept the Llama licence and use a token that can read it.
* `bitsandbytes ... no GPU` / CUDA OOM -> lower `--batch-size`, raise `--grad-accum`, keep `--max-length` >= 140.
* `SFTConfig got an unexpected keyword` -> TRL/Transformers API drift: install the pinned ranges (`transformers<5`, `trl<1`).
* Hub push 403 -> token needs *write* scope; set `HF_USERNAME`; or run with `--no-push` and upload `artifacts/<run>/final` yourself.
* Model outputs `0.00` / garbage -> you evaluated the base model or used a prompt without the trailing `Price is $`.
* W&B asks to log in on Colab -> set `WANDB_API_KEY` or pass `--no-wandb`.

## Verified / not verified
Verified (CPU, offline or small downloads): unit tests; `py_compile`; every CLI `--help`; `qlora-train --dry-run`; `SFTConfig`/`LoraConfig`
construction with `transformers 4.57.6` / `trl 0.29.1` / `peft 0.19.1`; and an end-to-end mini run (SFTTrainer prompt/completion training with
the project's settings -> adapter save -> `PeftModel` load -> `PricePredictor.generate/price`) on a tiny random Llama model on CPU.
**Not verified**: anything needing a CUDA GPU or gated models - 4-bit bitsandbytes loading, actual QLoRA training on Llama 3.2 3B, hub pushing,
W&B logging, tokenizer-based statistics on real data, and the reference benchmark numbers.

Design: [docs/lld-design.md](docs/lld-design.md).
