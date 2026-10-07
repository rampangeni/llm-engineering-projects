"""Import the Modal app definitions (no deployment, no network): checks decorators/wiring."""

import pytest

modal = pytest.importorskip("modal")


def test_class_service_defines_pricer():
    from price_model_service import service

    assert service.app.name == "pricer-service"
    assert service.Pricer is not None
    assert service.settings.min_containers == 0


def test_function_service_defines_price():
    from price_model_service import service_function

    assert hasattr(service_function.price, "remote")


def test_teaching_apps_import():
    from price_model_service.apps import hello, llama_smoke, pricer_ephemeral

    assert hasattr(hello.hello, "remote") and hasattr(hello.hello_europe, "remote")
    assert hasattr(llama_smoke.generate, "remote")
    assert hasattr(pricer_ephemeral.price, "remote")
