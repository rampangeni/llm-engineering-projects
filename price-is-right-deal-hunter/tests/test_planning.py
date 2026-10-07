import json

from conftest import FakePricer, fake_openai, make_selection, text_response, tool_response
from deal_hunter.agents.autonomous_planning import AutonomousPlanningAgent
from deal_hunter.agents.planning import PlanningAgent
from deal_hunter.config import Settings


class Scanner:
    def __init__(self, selection):
        self.selection, self.memory = selection, None

    def scan(self, memory):
        self.memory = memory
        return self.selection


class Ensemble:
    def __init__(self, mapping):
        self.mapping = mapping

    def price(self, description):
        return self.mapping[description]


class Messenger:
    def __init__(self):
        self.alerts, self.notifications = [], []

    def alert(self, opp):
        self.alerts.append(opp)

    def notify(self, *args):
        self.notifications.append(args)


def test_workflow_picks_best_discount_and_alerts():
    msg = Messenger()
    sel = make_selection(100, 100, 100)
    est = {"item 0": 120, "item 1": 400, "item 2": 90}
    p = PlanningAgent(Scanner(sel), Ensemble(est), msg, Settings())
    best = p.plan([])
    assert best.deal.url == "https://x/1" and best.discount == 300 and msg.alerts == [best]


def test_workflow_below_threshold_no_alert():
    msg = Messenger()
    p = PlanningAgent(Scanner(make_selection(100)), Ensemble({"item 0": 140}), msg, Settings(deal_threshold=50))
    assert p.plan([]) is None and msg.alerts == []


def test_workflow_limits_to_five_and_handles_no_selection():
    msg = Messenger()
    sel = make_selection(*[10] * 7)
    est = {f"item {i}": 10.0 for i in range(7)}
    seen = []

    class E(Ensemble):
        def price(self, d):
            seen.append(d)
            return 10.0

    PlanningAgent(Scanner(sel), E(est), msg, Settings()).plan([])
    assert len(seen) == 5
    assert PlanningAgent(Scanner(None), Ensemble({}), msg, Settings()).plan([]) is None


def auto(client, selection=None, est=None, **kw):
    msg = Messenger()
    agent = AutonomousPlanningAgent(Scanner(selection or make_selection(100)), Ensemble(est or {"item 0": 500}), msg,
                                    Settings(**kw), client=client)
    return agent, msg


def test_autonomous_tool_loop_end_to_end():
    notify_args = json.dumps({"description": "item 0", "deal_price": 100, "estimated_true_value": 500, "url": "https://x/0"})
    client = fake_openai(
        tool_response(("1", "scan_the_internet_for_bargains", "{}")),
        tool_response(("2", "estimate_true_value", json.dumps({"description": "item 0"}))),
        tool_response(("3", "notify_user_of_deal", notify_args)),
        text_response("OK"),
    )
    agent, msg = auto(client)
    opp = agent.plan([])
    assert opp.discount == 400 and opp.deal.url == "https://x/0" and len(msg.notifications) == 1
    # tool results were fed back to the model
    last_msgs = client.calls["create"][-1]["messages"]
    tool_msgs = [m for m in last_msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["1", "2", "3"]
    assert "item 0" in tool_msgs[0]["content"] and "500" in tool_msgs[1]["content"]
    assert len(client.calls["create"][0]["tools"]) == 3


def test_second_notification_is_ignored_and_plan_resets():
    args = json.dumps({"description": "d", "deal_price": 1, "estimated_true_value": 9, "url": "u"})
    client = fake_openai(tool_response(("1", "notify_user_of_deal", args), ("2", "notify_user_of_deal", args)), text_response("OK"))
    agent, msg = auto(client)
    assert agent.plan([]) is not None and len(msg.notifications) == 1
    client2 = fake_openai(text_response("nothing"))
    agent.openai = client2
    assert agent.plan([]) is None  # opportunity reset between runs


def test_unknown_tool_and_bad_args_become_tool_messages():
    client = fake_openai(
        tool_response(("1", "nope", "{}"), ("2", "estimate_true_value", "not json")), text_response("done")
    )
    agent, _ = auto(client)
    assert agent.plan([]) is None
    tool_msgs = [m for m in client.calls["create"][1]["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    assert "Unknown tool" in tool_msgs[0]["content"] and "failed" in tool_msgs[1]["content"]


def test_max_steps_guard():
    endless = [tool_response((str(i), "scan_the_internet_for_bargains", "{}")) for i in range(5)]
    agent, _ = auto(fake_openai(*endless), max_tool_steps=3)
    assert agent.plan([]) is None
    assert len(agent.openai.calls["create"]) == 3


def test_memory_passed_to_scanner():
    client = fake_openai(tool_response(("1", "scan_the_internet_for_bargains", "{}")), text_response("OK"))
    agent, _ = auto(client)
    agent.plan(["sentinel"])
    assert agent.scanner.memory == ["sentinel"]
