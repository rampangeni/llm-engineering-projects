import sys
import types

import pytest

from price_model_service.client import (MockPricer, ModalClassPricer, ModalFunctionPricer, PricerClient,
                                         make_backend)
from price_model_service.config import ServiceSettings


class Flaky:
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0

    def price(self, text):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return 42.0


def test_mock_is_deterministic_and_bounded():
    m = MockPricer()
    assert m.price("abc") == m.price("abc")
    assert 10 <= m.price("anything") < 510


def test_retry_then_success():
    sleeps = []
    c = PricerClient(backend=Flaky(2), retries=3, sleep=sleeps.append)
    assert c.price("x") == 42.0
    assert sleeps == [1, 2]


def test_retry_exhausted_raises():
    c = PricerClient(backend=Flaky(5), retries=2, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        c.price("x")


def test_preprocessor_applied_before_backend():
    seen = []

    class B:
        def price(self, t):
            seen.append(t)
            return 1.0

    class P:
        def preprocess(self, t):
            return "REWRITTEN:" + t

    PricerClient(backend=B(), preprocessor=P()).price("raw")
    assert seen == ["REWRITTEN:raw"]


@pytest.fixture
def fake_modal(monkeypatch):
    calls = {}

    class Method:
        def remote(self, text):
            calls["remote"] = text
            return 77

    class Instance:
        price = Method()

        def update_autoscaler(self, **kw):
            calls["autoscaler"] = kw

    class Cls:
        @staticmethod
        def from_name(app, name):
            calls["cls"] = (app, name)
            return lambda: Instance()

    class Function:
        @staticmethod
        def from_name(app, name):
            calls["fn"] = (app, name)
            return Method()

    mod = types.SimpleNamespace(Cls=Cls, Function=Function)
    monkeypatch.setitem(sys.modules, "modal", mod)
    return calls


def test_modal_class_backend_looks_up_by_name(fake_modal):
    s = ServiceSettings.from_env({"MODAL_APP_NAME": "my-app"})
    b = make_backend("modal-class", s)
    assert isinstance(b, ModalClassPricer)
    assert b.price("text") == 77.0
    assert fake_modal["cls"] == ("my-app", "Pricer")


def test_modal_function_backend(fake_modal):
    b = make_backend("modal-function", ServiceSettings())
    assert isinstance(b, ModalFunctionPricer)
    assert b.price("t") == 77.0 and fake_modal["fn"] == ("pricer-service", "price")


def test_keep_warm_and_cool(fake_modal):
    c = PricerClient(settings=ServiceSettings(), backend=make_backend("modal-class", ServiceSettings()))
    c.keep_warm(20)
    assert fake_modal["autoscaler"] == {"scaledown_window": 1200}
    c.cool_down()
    assert fake_modal["autoscaler"] == {"scaledown_window": 120}


def test_autoscaler_requires_class_backend():
    with pytest.raises(RuntimeError):
        PricerClient(backend=MockPricer()).keep_warm(5)


def test_unknown_backend():
    with pytest.raises(ValueError):
        make_backend("nope", ServiceSettings())
