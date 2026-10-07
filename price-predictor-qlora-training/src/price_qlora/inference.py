"""Inference with a base model + (optional) LoRA adapter from the Hub or a local folder.

This is the same recipe the week-8 Modal service uses:
base model in 4-bit NF4 (compute fp16) -> ``PeftModel.from_pretrained(base, adapter, revision=...)``
-> greedy ``generate(max_new_tokens=5)`` after the prompt ``...Price is $`` -> parse the number.
"""

from __future__ import annotations

from price_qlora import config
from price_qlora.parse import extract_price
from price_qlora.prompts import make_inference_prompt
from price_qlora.quant import load_base_model


class PricePredictor:
    """Callable ``predictor(prompt_or_description) -> str`` (generated continuation)."""

    def __init__(
        self,
        base_model: str | None = None,
        adapter: str | None = None,
        revision: str | None = None,
        quant: str = "4bit",
        compute_dtype: str = "float16",
        max_new_tokens: int = 5,
        seed: int = 42,
    ):
        import torch
        from transformers import set_seed

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        set_seed(seed)
        self.tokenizer, base = load_base_model(base_model or config.base_model(), quant, compute_dtype)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, adapter, revision=revision)
        else:
            self.model = base  # untuned base model (reference: 4-bit Llama 3.2 scored $110.72)
        self.model.eval()

    @classmethod
    def from_loaded(cls, tokenizer, model, max_new_tokens: int = 5, seed: int = 42) -> "PricePredictor":
        """Wrap an already-loaded tokenizer/model (e.g. the base model of qlora-quant-compare)."""
        import torch

        self = cls.__new__(cls)
        self.torch, self.max_new_tokens, self.seed = torch, max_new_tokens, seed
        self.tokenizer, self.model = tokenizer, model
        model.eval()
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def generate(self, prompt: str) -> str:
        """Only the newly generated text (e.g. '45.00')."""
        self.torch.manual_seed(self.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    def price(self, description: str, cutoff: int = config.CUTOFF) -> float:
        """Price in dollars for a free-text product description / summary."""
        prompt = make_inference_prompt(description, self.tokenizer, cutoff)
        return extract_price(self.generate(prompt))
