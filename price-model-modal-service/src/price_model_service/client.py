"""Thin client for the deployed pricer. Looks the Modal object up *by name* (no import of the
service module), so consumers - e.g. ../price-is-right-deal-hunter - only need the names.

Backends
--------
modal-class     modal.Cls.from_name(app, "Pricer")().price.remote(text)   (default, fast)
modal-function  modal.Function.from_name(app, "price").remote(text)       (slow, reloads model)
mock            deterministic offline stub, for development/tests without Modal or a GPU
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Protocol

from .config import ServiceSettings
from .preprocessor import Preprocessor

log = logging.getLogger(__name__)


class PricerBackend(Protocol):
    def price(self, description: str) -> float: ...


class MockPricer:
    """Offline stand-in: stable pseudo-price in [10, 510) derived from the text hash."""

    def price(self, description: str) -> float:
        digest = hashlib.md5(description.encode("utf-8")).hexdigest()
        return float(10 + int(digest[:6], 16) % 500)


class ModalClassPricer:
    def __init__(self, settings: ServiceSettings):
        import modal

        cls = modal.Cls.from_name(settings.app_name, settings.class_name)
        self._instance = cls()

    def price(self, description: str) -> float:
        return float(self._instance.price.remote(description))

    def set_scaledown_window(self, seconds: int) -> None:
        self._instance.update_autoscaler(scaledown_window=seconds)


class ModalFunctionPricer:
    def __init__(self, settings: ServiceSettings):
        import modal

        self._fn = modal.Function.from_name(settings.app_name, settings.function_name)

    def price(self, description: str) -> float:
        return float(self._fn.remote(description))


def make_backend(name: str, settings: ServiceSettings) -> PricerBackend:
    if name == "mock":
        return MockPricer()
    if name == "modal-class":
        return ModalClassPricer(settings)
    if name == "modal-function":
        return ModalFunctionPricer(settings)
    raise ValueError(f"Unknown backend {name!r}; use modal-class, modal-function or mock")


class PricerClient:
    """Retrying wrapper with optional rewrite step."""

    def __init__(
        self,
        backend: PricerBackend | None = None,
        settings: ServiceSettings | None = None,
        preprocessor: Preprocessor | None = None,
        retries: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings or ServiceSettings.from_env()
        self.backend = backend or make_backend(self.settings.backend, self.settings)
        self.preprocessor = preprocessor
        self.retries = max(1, retries if retries is not None else self.settings.retries)
        self._sleep = sleep

    def price(self, description: str) -> float:
        text = self.preprocessor.preprocess(description) if self.preprocessor else description
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return self.backend.price(text)
            except Exception as exc:  # noqa: BLE001 - remote errors are heterogeneous
                last = exc
                log.warning("pricer call failed (attempt %d/%d): %s", attempt, self.retries, exc)
                if attempt < self.retries:
                    self._sleep(2 ** (attempt - 1))
        raise RuntimeError(f"Pricer failed after {self.retries} attempts: {last}") from last

    def price_many(self, descriptions: list[str]) -> list[float]:
        return [self.price(d) for d in descriptions]

    def keep_warm(self, minutes: int = 20) -> None:
        """Stay warm for ``minutes`` instead of the default scaledown window (class backend)."""
        self._autoscale(minutes * 60)

    def cool_down(self) -> None:
        self._autoscale(self.settings.scaledown_window)

    def _autoscale(self, seconds: int) -> None:
        if not hasattr(self.backend, "set_scaledown_window"):
            raise RuntimeError("Autoscaler updates need the modal-class backend")
        self.backend.set_scaledown_window(seconds)
