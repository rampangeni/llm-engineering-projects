"""Names, defaults and the pure-logic training settings (no torch / transformers imports)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

DEFAULT_DATASET_USER = "rampangeni"  # private pre-built datasets: needs `hf auth login` or HF_TOKEN
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"  # W&B project and hub repo prefix: <user>/price-<RUN_NAME>

QUESTION = "What does this cost to the nearest dollar?"
PREFIX = "Price is $"

# Token budget for the *summary* inside a prompt; ~% of items truncated is printed by qlora-prepare-prompts.
CUTOFF = 110
# prompt = question (~12 tokens) + summary (<=110) + 'Price is $' (~4) + completion (~5) + BOS/EOS -> < 160
MAX_SEQ_LENGTH = 160


def base_model() -> str:
    return os.environ.get("BASE_MODEL", BASE_MODEL)


def dataset_user() -> str:
    return os.environ.get("DATASET_USERNAME", DEFAULT_DATASET_USER)


def hf_user() -> str | None:
    return os.environ.get("HF_USERNAME")


def items_dataset(lite: bool, user: str | None = None) -> str:
    """Summarised items dataset produced by price-dataset-and-baselines."""
    return f"{user or dataset_user()}/items_{'lite' if lite else 'full'}"


def prompts_dataset(lite: bool, user: str | None = None) -> str:
    """Prompt/completion dataset produced by qlora-prepare-prompts (splits: train, val, test)."""
    return f"{user or dataset_user()}/items_prompts_{'lite' if lite else 'full'}"


def make_run_name(now: datetime | None = None) -> str:
    """'2025-11-28_18.47.07' - same pattern as the public adapter repo 'price-2025-11-28_18.47.07'."""
    return (now or datetime.now()).strftime("%Y-%m-%d_%H.%M.%S")


def hub_model_id(user: str, run_name: str, project: str = PROJECT_NAME) -> str:
    return f"{user}/{project}-{run_name}"


def choose_compute_dtype(requested: str, bf16_supported: bool) -> str:
    """'auto' -> bfloat16 on Ampere+ GPUs, float16 otherwise (e.g. Colab T4)."""
    if requested == "auto":
        return "bfloat16" if bf16_supported else "float16"
    if requested not in ("bfloat16", "float16"):
        raise ValueError("compute dtype must be auto | bfloat16 | float16")
    return requested


@dataclass
class LoraSettings:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    def kwargs(self) -> dict:
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }


@dataclass
class TrainSettings:
    """All hyper-parameters of one run. ``sft_kwargs()`` / ``lora_kwargs()`` feed TRL / PEFT directly."""

    dataset: str = ""
    base_model: str = BASE_MODEL
    hf_user: str = ""
    run_name: str = ""
    project: str = PROJECT_NAME
    epochs: int = 1
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_32bit"
    max_length: int = MAX_SEQ_LENGTH
    eval_size: int = 1000  # validation items used for eval loss during training
    eval_steps: int = 2000
    save_steps: int = 2000
    logging_steps: int = 50
    seed: int = 42
    compute_dtype: str = "auto"  # auto | bfloat16 | float16
    wandb: bool = True
    push_to_hub: bool = True
    private_repo: bool = True
    output_dir: str = "artifacts"
    lora: LoraSettings = field(default_factory=LoraSettings)

    def __post_init__(self):
        self.run_name = self.run_name or make_run_name()

    @property
    def hub_id(self) -> str:
        return hub_model_id(self.hf_user, self.run_name, self.project)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, f"{self.project}-{self.run_name}")

    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def steps_per_epoch(self, n_train: int) -> int:
        return -(-n_train // self.effective_batch())  # ceil

    def adapt_to_dataset(self, n_train: int) -> bool:
        """Short runs (e.g. lite: ~1,250 steps) would never reach eval_steps/save_steps=2000 -> check ~4x per run.

        Returns True if eval/save intervals were changed.
        """
        total = self.steps_per_epoch(n_train) * self.epochs
        if total < max(self.eval_steps, self.save_steps):
            self.eval_steps = self.save_steps = max(total // 4, 1)
            return True
        return False

    def validate(self) -> None:
        if self.push_to_hub and not self.hf_user:
            raise ValueError("Set HF_USERNAME (or --hf-user) to push adapters, or pass --no-push")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch size and grad accumulation must be >= 1")
        if self.max_length < CUTOFF + 30:
            raise ValueError(f"max_length {self.max_length} is too small for CUTOFF {CUTOFF}: completions would be truncated")

    def lora_kwargs(self) -> dict:
        return self.lora.kwargs()

    def sft_kwargs(self, bf16: bool) -> dict:
        """kwargs for ``trl.SFTConfig`` (TRL >= 0.20 API, prompt/completion datasets)."""
        kw = {
            "output_dir": self.run_dir,
            "run_name": self.run_name,
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.grad_accum,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "optim": self.optim,
            "max_length": self.max_length,
            "completion_only_loss": True,  # loss only on the price tokens, not the prompt
            "bf16": bf16,
            "fp16": not bf16,
            "eval_strategy": "steps",
            "eval_steps": self.eval_steps,
            "save_strategy": "steps",
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "seed": self.seed,
            "report_to": "wandb" if self.wandb else "none",
            "group_by_length": False,
        }
        if self.push_to_hub:
            kw.update(
                push_to_hub=True,
                hub_model_id=self.hub_id,
                hub_strategy="every_save",
                hub_private_repo=self.private_repo,
            )
        return kw

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hub_id"] = self.hub_id if self.hf_user else None
        return d
