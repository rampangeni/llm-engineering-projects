"""Environment-driven settings shared by the Modal app definitions and the client.

This module must stay free of heavy imports (no modal/torch) so that it can be used
anywhere, including inside the Modal container image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Defaults reproduce a public pre-built adapter so the project runs out of the box.
# Override them with the adapter produced by ../price-predictor-qlora-training.
DEFAULT_HF_USER = "rampangeni"
DEFAULT_RUN_NAME = "2025-11-28_18.47.07"
DEFAULT_REVISION = "3e6de2528c7987fd54ac0c47094066f293655dc4"


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceSettings:
    app_name: str = "pricer-service"
    class_name: str = "Pricer"
    function_name: str = "price"
    hf_secret_name: str = "huggingface-secret"
    base_model: str = "meta-llama/Llama-3.2-3B"
    hf_user: str = DEFAULT_HF_USER
    project_name: str = "price"
    run_name: str = DEFAULT_RUN_NAME
    revision: str = DEFAULT_REVISION
    adapter_repo: str = ""
    base_only: bool = False
    gpu: str = "T4"
    timeout: int = 1800
    min_containers: int = 0
    scaledown_window: int = 120
    cache_volume: str = "hf-hub-cache"
    cache_dir: str = "/cache"
    backend: str = "modal-class"
    retries: int = 3

    @property
    def finetuned_model(self) -> str:
        """HF repo id of the LoRA adapter, or '' when serving the bare base model."""
        if self.base_only:
            return ""
        return self.adapter_repo or f"{self.hf_user}/{self.project_name}-{self.run_name}"

    @property
    def adapter_revision(self) -> str | None:
        return (self.revision or None) if self.finetuned_model else None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ServiceSettings":
        e = os.environ if env is None else env
        d = cls()
        return cls(
            app_name=e.get("MODAL_APP_NAME", d.app_name),
            class_name=e.get("MODAL_CLASS_NAME", d.class_name),
            function_name=e.get("MODAL_FUNCTION_NAME", d.function_name),
            hf_secret_name=e.get("HF_SECRET_NAME", d.hf_secret_name),
            base_model=e.get("BASE_MODEL", d.base_model),
            hf_user=e.get("HF_USER", d.hf_user),
            project_name=e.get("PROJECT_NAME", d.project_name),
            run_name=e.get("RUN_NAME", d.run_name),
            revision=e.get("ADAPTER_REVISION", d.revision),
            adapter_repo=e.get("ADAPTER_REPO", d.adapter_repo),
            base_only=_bool(e.get("BASE_ONLY")),
            gpu=e.get("GPU", d.gpu),
            timeout=int(e.get("TIMEOUT_SECONDS", d.timeout)),
            min_containers=int(e.get("MIN_CONTAINERS", d.min_containers)),
            scaledown_window=int(e.get("SCALEDOWN_WINDOW_SECONDS", d.scaledown_window)),
            cache_volume=e.get("HF_CACHE_VOLUME", d.cache_volume),
            backend=e.get("PRICER_BACKEND", d.backend),
            retries=int(e.get("CLIENT_RETRIES", d.retries)),
        )

    def container_env(self) -> dict[str, str]:
        """Settings forwarded into the Modal container (the container has no .env file)."""
        return {
            "BASE_MODEL": self.base_model,
            "ADAPTER_REPO": self.finetuned_model,
            "ADAPTER_REVISION": self.revision,
            "BASE_ONLY": "1" if not self.finetuned_model else "0",
            "HF_HUB_CACHE": self.cache_dir,
        }
