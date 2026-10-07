from datetime import datetime

import pytest

from price_qlora import config
from price_qlora.config import LoraSettings, TrainSettings
from price_qlora.quant import bnb_kwargs, torch_dtype_name
from price_qlora.train import build_parser, settings_from_args


def test_names(monkeypatch):
    monkeypatch.setenv("DATASET_USERNAME", "ds")
    assert config.prompts_dataset(True) == "ds/items_prompts_lite"
    assert config.items_dataset(False) == "ds/items_full"
    assert config.prompts_dataset(False, user="me") == "me/items_prompts_full"
    run = config.make_run_name(datetime(2025, 11, 28, 18, 47, 7))
    assert run == "2025-11-28_18.47.07"
    assert config.hub_model_id("rampangeni", run) == "rampangeni/price-2025-11-28_18.47.07"


def test_dtype_choice():
    assert config.choose_compute_dtype("auto", True) == "bfloat16"
    assert config.choose_compute_dtype("auto", False) == "float16"
    assert config.choose_compute_dtype("float16", True) == "float16"
    with pytest.raises(ValueError):
        config.choose_compute_dtype("int4", True)


def test_lora_kwargs_defaults():
    kw = LoraSettings().kwargs()
    assert kw["r"] == 32 and kw["lora_alpha"] == 64 and kw["lora_dropout"] == 0.1
    assert kw["target_modules"] == ["q_proj", "v_proj", "k_proj", "o_proj"] and kw["task_type"] == "CAUSAL_LM"


def test_sft_kwargs_push_and_precision():
    s = TrainSettings(hf_user="me", run_name="r1", dataset="d")
    kw = s.sft_kwargs(bf16=True)
    assert kw["hub_model_id"] == "me/price-r1" and kw["hub_strategy"] == "every_save" and kw["hub_private_repo"] is True
    assert kw["bf16"] and not kw["fp16"] and kw["completion_only_loss"] is True
    assert kw["output_dir"].endswith("price-r1") and kw["report_to"] == "wandb"
    kw2 = TrainSettings(push_to_hub=False, wandb=False, run_name="x").sft_kwargs(bf16=False)
    assert "hub_model_id" not in kw2 and kw2["report_to"] == "none" and kw2["fp16"]


def test_effective_batch_and_steps():
    s = TrainSettings(batch_size=4, grad_accum=4)
    assert s.effective_batch() == 16
    assert s.steps_per_epoch(800_000) == 50_000
    assert s.steps_per_epoch(20_001) == 1251


def test_validate():
    with pytest.raises(ValueError):
        TrainSettings(hf_user="").validate()  # push enabled but no user
    TrainSettings(hf_user="", push_to_hub=False).validate()
    with pytest.raises(ValueError):
        TrainSettings(push_to_hub=False, max_length=64).validate()


def test_quant_presets():
    k = bnb_kwargs("4bit", "float16")
    assert k == {"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float16", "bnb_4bit_quant_type": "nf4"}
    assert bnb_kwargs("8bit") == {"load_in_8bit": True}
    assert bnb_kwargs("bf16") is None
    assert torch_dtype_name("bf16") == "bfloat16" and torch_dtype_name("8bit") is None
    with pytest.raises(ValueError):
        bnb_kwargs("2bit")


def test_cli_to_settings(monkeypatch):
    monkeypatch.setenv("HF_USERNAME", "me")
    monkeypatch.delenv("BASE_MODEL", raising=False)
    monkeypatch.setenv("DATASET_USERNAME", "rampangeni")
    args = build_parser().parse_args(["--lite", "--batch-size", "4", "--grad-accum", "4", "--no-wandb", "--lora-r", "16"])
    s = settings_from_args(args)
    assert s.dataset == "rampangeni/items_prompts_lite" and s.hf_user == "me"
    assert s.base_model == "meta-llama/Llama-3.2-3B" and s.effective_batch() == 16
    assert s.lora.r == 16 and s.wandb is False
    assert s.to_dict()["hub_id"].startswith("me/price-")


def test_dry_run_needs_no_gpu(monkeypatch, capsys):
    from price_qlora import train

    monkeypatch.setenv("HF_USERNAME", "me")
    train.main(["--lite", "--dry-run"])
    out = capsys.readouterr().out
    assert "Dry run" in out and '"learning_rate": 0.0001' in out


def test_adapt_to_dataset_for_short_runs():
    s = TrainSettings(batch_size=16)
    assert s.adapt_to_dataset(20_000) is True  # 1,250 steps < 2,000
    assert s.eval_steps == s.save_steps == 312
    big = TrainSettings(batch_size=16)
    assert big.adapt_to_dataset(800_000) is False and big.eval_steps == 2000
