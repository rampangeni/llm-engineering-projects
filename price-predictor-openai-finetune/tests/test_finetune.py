from types import SimpleNamespace

import pytest

from price_openai_ft import config, finetune


def test_build_job_kwargs_defaults_are_expected():
    kw = finetune.build_job_kwargs("file-train", "file-val")
    assert kw == {
        "training_file": "file-train",
        "validation_file": "file-val",
        "model": "gpt-4.1-nano-2025-04-14",
        "seed": 42,
        "hyperparameters": {"n_epochs": 1, "batch_size": 1},
        "suffix": "pricer",
    }


def test_build_job_kwargs_wandb_and_lr():
    kw = finetune.build_job_kwargs("t", None, learning_rate_multiplier=0.5, wandb_project="p", wandb_run_name="r", batch_size="auto")
    assert "validation_file" not in kw
    assert kw["hyperparameters"] == {"n_epochs": 1, "batch_size": "auto", "learning_rate_multiplier": 0.5}
    assert kw["integrations"] == [{"type": "wandb", "wandb": {"project": "p", "name": "r"}}]


class FakeClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.created = None
        self.files = SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id=f"{purpose}-{file.read()[:3].decode()}"))
        self.fine_tuning = SimpleNamespace(
            jobs=SimpleNamespace(create=self._create, retrieve=self._retrieve, list_events=self._events, list=self._list)
        )
        self.calls = 0

    def _create(self, **kw):
        self.created = kw
        return SimpleNamespace(id="ftjob-1")

    def _retrieve(self, job_id):
        st = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return SimpleNamespace(id=job_id, status=st, model="m", fine_tuned_model="ft:gpt:me::abc" if st == "succeeded" else None, error=None)

    def _events(self, fine_tuning_job_id, limit):
        return SimpleNamespace(data=[SimpleNamespace(message=f"event while {self.calls}")])

    def _list(self, limit):
        return SimpleNamespace(data=[SimpleNamespace(id="ftjob-latest")])


def test_upload_uses_fine_tune_purpose(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("abc\n")
    assert finetune.upload(FakeClient([]), f) == "fine-tune-abc"


def test_launch_and_wait():
    c = FakeClient(["validating_files", "running", "succeeded"])
    job = finetune.launch(c, train_file_id="t", validation_file_id="v")
    assert job.id == "ftjob-1" and c.created["training_file"] == "t"
    printed = []
    s = finetune.wait(c, "ftjob-1", poll_seconds=0, printer=printed.append, sleep=lambda s: None)
    assert s["status"] == "succeeded" and s["fine_tuned_model"] == "ft:gpt:me::abc"
    assert printed  # events were reported, each only once
    assert len(printed) == len(set(printed))


def test_wait_returns_on_failure():
    c = FakeClient(["running", "failed"])
    assert finetune.wait(c, "j", sleep=lambda s: None, printer=lambda m: None)["status"] == "failed"


def test_latest_job_id():
    assert finetune.latest_job_id(FakeClient([])) == "ftjob-latest"


def test_state_roundtrip_merges(tmp_path):
    p = tmp_path / "a" / "state.json"
    finetune.save_state(p, job_id="j1", train_file_id="t")
    finetune.save_state(p, fine_tuned_model="ft:x", job_id=None)  # None must not erase job_id
    assert finetune.load_state(p) == {"job_id": "j1", "train_file_id": "t", "fine_tuned_model": "ft:x"}


def test_dataset_name(monkeypatch):
    monkeypatch.setenv("HF_USERNAME", "me")
    assert config.dataset_name(False) == "me/items_full"
    assert config.dataset_name(True) == "me/items_lite"
