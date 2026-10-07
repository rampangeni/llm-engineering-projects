"""Model loading + generation, executed INSIDE the Modal container.

All heavy imports are lazy so this module can be imported locally without torch.
It is shipped to the container with ``Image.add_local_python_source``.
"""

from __future__ import annotations

from .config import ServiceSettings
from .prompts import build_prompt, extract_price


def load_model(settings: ServiceSettings):
    """Load tokenizer + 4-bit quantised base model (+ LoRA adapter unless base-only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model, quantization_config=quant_config, device_map="auto"
    )
    if settings.finetuned_model:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, settings.finetuned_model, revision=settings.adapter_revision
        )
    return tokenizer, model


def generate_price(tokenizer, model, description: str, max_new_tokens: int = 5) -> float:
    import torch
    from transformers import set_seed

    set_seed(42)
    inputs = tokenizer.encode(build_prompt(description), return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
    continuation = tokenizer.decode(outputs[0][inputs.shape[1]:])
    return extract_price(continuation)
