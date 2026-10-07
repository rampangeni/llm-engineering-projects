"""Upload files, launch / monitor the OpenAI fine-tuning job. All functions take an ``openai`` client
so they can be mocked in tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from price_openai_ft import config

TERMINAL = {"succeeded", "failed", "cancelled"}


def upload(client, path: str | Path) -> str:
    """Upload a jsonl file for fine-tuning -> file id."""
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose="fine-tune").id


def build_job_kwargs(
    train_file_id: str,
    validation_file_id: str | None = None,
    model: str = config.BASE_MODEL,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int | str = config.DEFAULT_BATCH_SIZE,
    learning_rate_multiplier: float | str | None = None,
    suffix: str = config.SUFFIX,
    seed: int = config.SEED,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> dict:
    hyper: dict = {"n_epochs": epochs, "batch_size": batch_size}
    if learning_rate_multiplier is not None:
        hyper["learning_rate_multiplier"] = learning_rate_multiplier
    kwargs: dict = {
        "training_file": train_file_id,
        "model": model,
        "seed": seed,
        "hyperparameters": hyper,
        "suffix": suffix,
    }
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    if wandb_project:
        wandb = {"project": wandb_project}
        if wandb_run_name:
            wandb["name"] = wandb_run_name
        kwargs["integrations"] = [{"type": "wandb", "wandb": wandb}]
    return kwargs


def launch(client, **kwargs):
    """Create the job. Returns the job object."""
    return client.fine_tuning.jobs.create(**build_job_kwargs(**kwargs))


def status(client, job_id: str, n_events: int = 10) -> dict:
    job = client.fine_tuning.jobs.retrieve(job_id)
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=n_events).data
    return {
        "id": job.id,
        "status": job.status,
        "model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": getattr(job, "trained_tokens", None),
        "error": getattr(job, "error", None),
        "events": [e.message for e in reversed(events)],
    }


def wait(client, job_id: str, poll_seconds: int = 30, printer=print, sleep=time.sleep) -> dict:
    """Poll until the job reaches succeeded / failed / cancelled."""
    seen: set[str] = set()
    while True:
        s = status(client, job_id, n_events=20)
        for msg in s["events"]:
            if msg not in seen:
                seen.add(msg)
                printer(msg)
        if s["status"] in TERMINAL:
            return s
        sleep(poll_seconds)


def latest_job_id(client) -> str:
    return client.fine_tuning.jobs.list(limit=1).data[0].id


def save_state(path: str | Path, **fields) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_state(path) if path.exists() else {}
    current.update({k: v for k, v in fields.items() if v is not None})
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def load_state(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
