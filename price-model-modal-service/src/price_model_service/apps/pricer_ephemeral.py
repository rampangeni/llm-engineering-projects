"""Step 3: ephemeral pricer app.

Runs only while ``with app.run():`` is active; every call re-loads the model, so it is slow.
Good for a first end-to-end check of GPU + adapter + secret before deploying.
"""

import modal
from dotenv import load_dotenv

from price_model_service.config import ServiceSettings

load_dotenv()
settings = ServiceSettings.from_env()

app = modal.App("pricer")  # distinct from the deployed app name on purpose
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


@app.local_entrypoint()
def main(description: str = "Quadcast HyperX condenser mic, connects via usb-c to your computer"):
    print(price.remote(description))
