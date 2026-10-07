"""Step 2: run the *base* Llama 3.2 3B on a T4 to prove GPU + HF secret work."""

import modal
from dotenv import load_dotenv

from price_model_service.config import ServiceSettings

load_dotenv()
settings = ServiceSettings.from_env()

app = modal.App("llama")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate")
secrets = [modal.Secret.from_name(settings.hf_secret_name)]


@app.function(image=image, secrets=secrets, gpu=settings.gpu, timeout=settings.timeout)
def generate(prompt: str, model_name: str = settings.base_model) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0])


@app.local_entrypoint()
def main(prompt: str = "Never gonna give you up, never gonna"):
    print(generate.remote(prompt))
