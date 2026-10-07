import pytest

from deal_hunter.config import Settings, parse_weights
from deal_hunter.models import Deal, DealSelection, Opportunity


def test_opportunity_roundtrip():
    o = Opportunity(deal=Deal(product_description="d", price=10, url="u"), estimate=30, discount=20)
    assert Opportunity(**o.model_dump()) == o


def test_deal_selection_schema_has_descriptions():
    schema = DealSelection.model_json_schema()
    assert "deals" in schema["properties"]


def test_parse_weights_ok_and_errors():
    assert parse_weights("frontier:0.5, nn:0.5") == {"frontier": 0.5, "nn": 0.5}
    with pytest.raises(ValueError):
        parse_weights("bogus:1")
    with pytest.raises(ValueError):
        parse_weights("frontier:-1")


def test_settings_from_env():
    s = Settings.from_env({"DEAL_THRESHOLD": "75", "ENABLE_NN": "0", "PLANNER": "autonomous",
                           "DEAL_FEEDS": "a, b", "ENSEMBLE_WEIGHTS": "frontier:1"})
    assert s.deal_threshold == 75 and not s.enable_nn and s.enable_frontier
    assert s.planner == "autonomous" and s.feeds == ("a", "b") and s.ensemble_weights == {"frontier": 1.0}


def test_settings_defaults():
    s = Settings.from_env({})
    assert s.ensemble_weights == {"frontier": 0.8, "specialist": 0.1, "nn": 0.1}
    assert s.modal_app_name == "pricer-service" and s.modal_class_name == "Pricer"
    assert len(s.feeds) == 3
