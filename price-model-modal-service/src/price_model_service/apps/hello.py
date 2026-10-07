"""Step 1: the smallest possible Modal app. No GPU, no secrets."""

import modal

app = modal.App("hello")
image = modal.Image.debian_slim().pip_install("requests")


def _where() -> str:
    import requests

    data = requests.get("https://ipinfo.io/json", timeout=10).json()
    return f"Hello from {data['city']}, {data['region']}, {data['country']}!!"


@app.function(image=image)
def hello() -> str:
    return _where()


@app.function(image=image, region="eu")
def hello_europe() -> str:
    """Region pinning (costs slightly more credits)."""
    return _where()


@app.local_entrypoint()
def main():
    print(hello.remote())
