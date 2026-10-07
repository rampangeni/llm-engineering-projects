import json

import pytest

from price_openai_ft.items import Item
from price_openai_ft.prompts import INSTRUCTION, make_jsonl, messages_for, test_messages_for as inference_messages, validate_jsonl, write_jsonl


def item(i=0, price=12.3456, summary="Title: Foo\nCategory: Bar"):
    return Item(title=f"t{i}", category="c", price=price, summary=summary)


def test_messages_for_matches_expected_format():
    msgs = messages_for(item())
    assert msgs[0] == {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: Foo\nCategory: Bar"}
    assert msgs[1] == {"role": "assistant", "content": "$12.35"}


def test_inference_prompt_is_training_prompt_minus_answer():
    assert inference_messages(item()) == messages_for(item())[:1]


def test_price_formatting():
    assert messages_for(item(price=5))[1]["content"] == "$5.00"
    assert messages_for(item(price=999.49))[1]["content"] == "$999.49"


def test_make_jsonl_lines_are_valid_json():
    out = make_jsonl([item(1), item(2, price=7)])
    lines = out.split("\n")
    assert len(lines) == 2 and not out.endswith("\n")
    obj = json.loads(lines[1])
    assert list(obj) == ["messages"] and obj["messages"][1]["content"] == "$7.00"


def test_summary_with_newlines_stays_on_one_line():
    assert "\n" not in make_jsonl([item(summary="a\nb\nc")]).strip("\n").replace("\\n", "")


def test_write_and_validate(tmp_path):
    p = write_jsonl([item(i) for i in range(12)], tmp_path / "sub" / "t.jsonl")
    assert validate_jsonl(p) == 12


def test_validate_rejects_bad_file(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")
    with pytest.raises(ValueError):
        validate_jsonl(p)
    p.write_text(json.dumps({"messages": [{"role": "assistant", "content": "12"}]}) + "\n")
    with pytest.raises(ValueError):
        validate_jsonl(p)
