from types import SimpleNamespace

import pytest

from price_openai_ft import cli, finetune
from price_openai_ft.evaluator import Tester, color_for, compute_metrics, post_process
from price_openai_ft.items import Item
from price_openai_ft.predictor import make_predictor


def test_post_process_and_colors():
    assert post_process("$1,234.50") == 1234.5
    assert post_process("n/a") == 0
    assert color_for(10, 500) == "green" and color_for(70, 200) == "orange" and color_for(300, 400) == "red"


def test_metrics_values():
    m = compute_metrics([100, 100], [100, 200])
    assert m.average_error == 50 and m.hit_rate == 0.5 and m.red == 1 and m.green == 1
    assert m.mse == 5000


def test_predictor_calls_chat_with_max_tokens():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="$99.00"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    pred = make_predictor(client, "ft:gpt-4.1-nano:me:pricer:abc")
    item = Item(title="t", category="c", price=100.0, summary="S")
    assert pred(item) == "$99.00"
    assert seen["max_tokens"] == 7 and seen["model"].startswith("ft:")
    assert seen["messages"] == [{"role": "user", "content": seen["messages"][0]["content"]}]
    t = Tester(pred, [item] * 3, size=10, workers=2, quiet=True)
    m = t.run()
    assert m.n == 3 and m.average_error == 1 and t.rows()[0]["raw"] == "$99.00"


def test_resolve_model(tmp_path):
    st = tmp_path / "s.json"
    assert cli.resolve_model("base", str(st)) == "gpt-4.1-nano"
    assert cli.resolve_model("custom-model", str(st)) == "custom-model"
    with pytest.raises(SystemExit):
        cli.resolve_model("ft", str(st))
    finetune.save_state(st, fine_tuned_model="ft:abc")
    assert cli.resolve_model("ft", str(st)) == "ft:abc"


def test_parser_defaults():
    a = cli.build_parser().parse_args(["launch"])
    assert a.epochs == 1 and a.batch_size == "1" and a.suffix == "pricer"
    a = cli.build_parser().parse_args(["--lite", "prepare"])
    assert a.train == 100 and a.val == 50 and a.lite


def test_prepare_end_to_end_with_stubbed_hub(tmp_path, monkeypatch):
    items = [Item(title=f"t{i}", category="c", price=10.0 + i, summary=f"S{i}") for i in range(20)]
    monkeypatch.setattr(cli.Item, "from_hub", classmethod(lambda cls, name: (items, items[:10], items[:5])))
    monkeypatch.setattr(cli.config, "hf_login", lambda: None)
    cli.main(["prepare", "--train", "15", "--val", "10", "--out-dir", str(tmp_path)])
    assert len((tmp_path / "fine_tune_train.jsonl").read_text().splitlines()) == 15
    with pytest.raises(SystemExit):
        cli.main(["prepare", "--train", "5", "--out-dir", str(tmp_path)])  # < OpenAI minimum
