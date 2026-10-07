import pytest

from conftest import FakePricer
from deal_hunter.agents.ensemble import EnsembleAgent
from deal_hunter.config import Settings

W = {"frontier": 0.8, "specialist": 0.1, "nn": 0.1}


def comps(f=100.0, s=200.0, n=300.0):
    return {"frontier": FakePricer(f), "specialist": FakePricer(s), "nn": FakePricer(n)}


def test_default_weighting_matches_reference_formula():
    e = EnsembleAgent(comps(), W)
    assert e.price("x") == pytest.approx(100 * 0.8 + 200 * 0.1 + 300 * 0.1)


def test_min_strategy():
    assert EnsembleAgent(comps(), W, strategy="min").price("x") == 100.0


def test_failed_component_renormalises_weights():
    c = comps()
    c["nn"] = FakePricer(fail=True)
    expected = (100 * 0.8 + 200 * 0.1) / 0.9
    assert EnsembleAgent(c, W).price("x") == pytest.approx(expected)


def test_all_fail_raises():
    with pytest.raises(RuntimeError):
        EnsembleAgent({"frontier": FakePricer(fail=True)}, W).price("x")


def test_unweighted_survivor_uses_mean():
    e = EnsembleAgent({"frontier": FakePricer(10), "nn": FakePricer(30)}, {"specialist": 1.0})
    assert e.price("x") == 20


def test_preprocessor_output_is_what_components_see():
    class P:
        model_name = "fake"

        def preprocess(self, t):
            return "REWRITE"

    c = comps()
    EnsembleAgent(c, W, preprocessor=P()).price("raw")
    assert all(a.seen == ["REWRITE"] for a in c.values())


def test_requires_components_and_valid_strategy():
    with pytest.raises(ValueError):
        EnsembleAgent({})
    with pytest.raises(ValueError):
        EnsembleAgent(comps(), strategy="avg")


def test_from_settings_skips_disabled_and_failing():
    s = Settings(enable_preprocessor=False, enable_nn=True)

    def boom():
        raise FileNotFoundError("no weights")

    e = EnsembleAgent.from_settings(
        object(), s, factories={"specialist": lambda: FakePricer(50), "frontier": lambda: FakePricer(70), "nn": boom}
    )
    assert set(e.components) == {"specialist", "frontier"} and e.preprocessor is None
    e2 = EnsembleAgent.from_settings(None, s, factories={"frontier": lambda: FakePricer(1), "specialist": lambda: FakePricer(2)})
    assert set(e2.components) == {"specialist"}  # frontier needs a collection
