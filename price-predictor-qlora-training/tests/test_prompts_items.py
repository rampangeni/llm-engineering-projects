from conftest import make_item
from price_qlora.config import CUTOFF, PREFIX, QUESTION
from price_qlora.prompts import build_prompt_splits, make_inference_prompt, truncation_stats


def test_make_prompts_train_rounds(tok):
    it = make_item(price=49.6, words=3)
    it.make_prompts(tok, 110, True)
    assert it.prompt == f"{QUESTION}\n\nw0 w1 w2\n\n{PREFIX}"
    assert it.completion == "50.00"
    assert it.to_datapoint() == {"prompt": it.prompt, "completion": "50.00"}


def test_make_prompts_test_keeps_exact_price(tok):
    it = make_item(price=49.6)
    it.make_prompts(tok, 110, False)
    assert it.completion == "49.6"


def test_truncation_to_cutoff(tok):
    it = make_item(words=20)
    it.make_prompts(tok, 5, True)
    assert "w4" in it.prompt and "w5" not in it.prompt
    assert it.count_tokens(tok) == 20


def test_prompt_ends_with_prefix_so_model_generates_the_number(tok):
    it = make_item()
    it.make_prompts(tok, CUTOFF, True)
    assert it.prompt.endswith("Price is $") and not it.completion.startswith("$")


def test_count_prompt_tokens(tok):
    it = make_item(words=2)
    it.make_prompts(tok, 110, True)
    assert it.count_prompt_tokens(tok) == len((it.prompt + it.completion).split(" "))


def test_build_prompt_splits_rounding_differs_by_split(tok):
    tr, va, te = [make_item(1)], [make_item(2)], [make_item(3, price=12.34)]
    build_prompt_splits(tr, va, te, tok)
    assert tr[0].completion == "50.00" and va[0].completion == "50.00" and te[0].completion == "12.34"


def test_inference_prompt(tok):
    assert make_inference_prompt("a b c") == f"{QUESTION}\n\na b c\n\n{PREFIX}"
    assert make_inference_prompt("a b c d", tok, cutoff=2) == f"{QUESTION}\n\na b\n\n{PREFIX}"


def test_truncation_stats(tok):
    items = [make_item(words=n) for n in (5, 10, 200)]
    s = truncation_stats(items, tok, cutoff=110)
    assert s["truncated"] == 1 and s["max_tokens"] == 200 and abs(s["truncated_fraction"] - 1 / 3) < 1e-9


def test_item_ignores_extra_columns():
    from price_qlora.items import Item

    assert Item.model_validate({"title": "t", "category": "c", "price": 1, "foo": 1}).price == 1
