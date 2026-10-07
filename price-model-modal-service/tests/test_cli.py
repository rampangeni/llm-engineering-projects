import json

from price_model_service import cli


def test_config_command(capsys):
    assert cli.main(["config"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["finetuned_model"].startswith("rampangeni/price-")


def test_deploy_dry_run(capsys):
    assert cli.main(["deploy", "--dry-run", "--variant", "function"]) == 0
    assert "price_model_service.service_function" in capsys.readouterr().out


def test_price_mock_backend(capsys):
    assert cli.main(["price", "--backend", "mock", "an iPhone"]) == 0
    assert capsys.readouterr().out.startswith("$")
