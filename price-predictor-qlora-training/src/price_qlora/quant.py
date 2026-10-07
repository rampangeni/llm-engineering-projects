"""Quantisation presets for loading the base model (4-bit vs 8-bit vs 16-bit)."""

from __future__ import annotations

PRESETS = ("4bit", "8bit", "bf16", "fp16")


def bnb_kwargs(preset: str, compute_dtype: str = "bfloat16") -> dict | None:
    """kwargs for ``BitsAndBytesConfig`` (dtype given by name), or None for unquantised presets."""
    if preset == "4bit":
        # QLoRA: NF4 weights + double quantisation (quantise the quantisation constants too)
        return {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_quant_type": "nf4",
        }
    if preset == "8bit":
        return {"load_in_8bit": True}
    if preset in ("bf16", "fp16"):
        return None
    raise ValueError(f"unknown quantisation preset {preset!r}; choose from {PRESETS}")


def torch_dtype_name(preset: str, compute_dtype: str = "bfloat16") -> str | None:
    """dtype for the non-quantised parts when loading (``torch_dtype=``)."""
    return {"bf16": "bfloat16", "fp16": "float16"}.get(preset, compute_dtype if preset == "4bit" else None)


def make_quant_config(preset: str, compute_dtype: str = "bfloat16"):
    """-> transformers.BitsAndBytesConfig | None (needs the ``gpu`` extra)."""
    import torch
    from transformers import BitsAndBytesConfig

    kw = bnb_kwargs(preset, compute_dtype)
    if kw is None:
        return None
    if "bnb_4bit_compute_dtype" in kw:
        kw["bnb_4bit_compute_dtype"] = getattr(torch, kw["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**kw)


def load_base_model(base_model: str, preset: str = "4bit", compute_dtype: str = "bfloat16"):
    """Load tokenizer + base model with the chosen preset (device_map='auto')."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"device_map": "auto"}
    qc = make_quant_config(preset, compute_dtype)
    if qc is not None:
        kwargs["quantization_config"] = qc
    else:
        kwargs["torch_dtype"] = getattr(torch, torch_dtype_name(preset, compute_dtype))
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def footprint_mb(model) -> float:
    return model.get_memory_footprint() / 1e6
