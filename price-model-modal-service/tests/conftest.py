import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ["MODAL_APP_NAME", "ADAPTER_REPO", "BASE_ONLY", "HF_USER", "RUN_NAME", "ADAPTER_REVISION",
              "PRICER_BACKEND", "MIN_CONTAINERS", "PRICER_PREPROCESSOR_MODEL"]:
        monkeypatch.delenv(k, raising=False)
