"""Deployed Modal app, function flavour: reloads the model on every call (slow, simple).

Deploy:  modal deploy -m price_model_service.service_function
Call:    modal.Function.from_name("pricer-service", "price").remote("...")

NOTE: it uses the same app name as ``service`` by
default, so deploying one replaces the other. Prefer ``service`` (class) in practice.
"""

from __future__ import annotations

import modal
from dotenv import load_dotenv

from .config import ServiceSettings

load_dotenv()
settings = ServiceSettings.from_env()

app = modal.App(settings.app_name)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
    .env(settings.container_env())
    .add_local_python_source("price_model_service")
)
secrets = [modal.Secret.from_name(settings.hf_secret_name)]


@app.function(image=image, secrets=secrets, gpu=settings.gpu, timeout=settings.timeout)
def price(description: str) -> float:
    from price_model_service._inference import generate_price, load_model
    from price_model_service.config import ServiceSettings as S

    tokenizer, model = load_model(S.from_env())
    return generate_price(tokenizer, model, description)
