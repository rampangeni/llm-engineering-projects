from price_model_service.config import ServiceSettings


def test_defaults_reproduce_public_adapter():
    s = ServiceSettings.from_env({})
    assert s.finetuned_model == "rampangeni/price-2025-11-28_18.47.07"
    assert s.adapter_revision.startswith("3e6de25")
    assert s.app_name == "pricer-service" and s.min_containers == 0


def test_env_overrides_and_adapter_repo_wins():
    s = ServiceSettings.from_env(
        {"HF_USER": "me", "RUN_NAME": "r1", "MIN_CONTAINERS": "1", "GPU": "L4"}
    )
    assert s.finetuned_model == "me/price-r1" and s.min_containers == 1 and s.gpu == "L4"
    s2 = ServiceSettings.from_env({"ADAPTER_REPO": "x/y", "HF_USER": "me"})
    assert s2.finetuned_model == "x/y"


def test_base_only_fallback_has_no_adapter():
    s = ServiceSettings.from_env({"BASE_ONLY": "1"})
    assert s.finetuned_model == "" and s.adapter_revision is None
    assert s.container_env()["BASE_ONLY"] == "1"


def test_container_env_roundtrip():
    s = ServiceSettings.from_env({"ADAPTER_REPO": "x/y", "ADAPTER_REVISION": "abc"})
    s2 = ServiceSettings.from_env(s.container_env())
    assert s2.finetuned_model == "x/y" and s2.revision == "abc"
