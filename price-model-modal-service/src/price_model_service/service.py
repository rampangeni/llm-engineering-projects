"""Deployed Modal app: class-based pricer with model cached in memory across calls.

Deploy:   modal deploy -m price_model_service.service     (or: price-model deploy)
Call:     modal.Cls.from_name("pricer-service", "Pricer")().price.remote("...")

The model is loaded once per container in
``@modal.enter()``; weights are cached on a Modal Volume so cold starts skip the download.
"""

from __future__ import annotations

import modal
from dotenv import load_dotenv

from .config import ServiceSettings

load_dotenv()  # local only; the container receives settings via image.env()
settings = ServiceSettings.from_env()

app = modal.App(settings.app_name)

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "torch", "transformers", "bitsandbytes", "accelerate", "peft")
    .env(settings.container_env())
    .add_local_python_source("price_model_service")
)
secrets = [modal.Secret.from_name(settings.hf_secret_name)]
hf_cache_volume = modal.Volume.from_name(settings.cache_volume, create_if_missing=True)


@app.cls(
    image=image,
    secrets=secrets,
    gpu=settings.gpu,
    timeout=settings.timeout,
    min_containers=settings.min_containers,
    scaledown_window=settings.scaledown_window,
    volumes={settings.cache_dir: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        from price_model_service._inference import load_model
        from price_model_service.config import ServiceSettings as S

        self.tokenizer, self.model = load_model(S.from_env())

    @modal.method()
    def price(self, description: str) -> float:
        from price_model_service._inference import generate_price

        return generate_price(self.tokenizer, self.model, description)
