import json

from conftest import make_deal
from deal_hunter.framework import DealAgentFramework, build_planner, load_seed_memory
from deal_hunter.models import Opportunity
from deal_hunter.agents.planning import PlanningAgent


def opp(url="https://x/9"):
    return Opportunity(deal=make_deal(url), estimate=500, discount=400)


class StubPlanner:
    def __init__(self, result):
        self.result, self.memory_seen = result, None

    def plan(self, memory):
        self.memory_seen = list(memory)
        return self.result


def test_missing_memory_starts_empty(settings):
    assert DealAgentFramework(settings, planner=StubPlanner(None)).memory == []


def test_run_appends_and_persists(settings):
    fw = DealAgentFramework(settings, planner=StubPlanner(opp()))
    fw.run()
    saved = json.load(open(settings.memory_path))
    assert saved[0]["deal"]["url"] == "https://x/9"
    assert len(DealAgentFramework(settings, planner=StubPlanner(None)).memory) == 1


def test_run_without_result_does_not_write(settings):
    fw = DealAgentFramework(settings, planner=StubPlanner(None))
    fw.run()
    import os

    assert not os.path.exists(settings.memory_path)


def test_reset_memory_truncates_and_seeds(settings):
    DealAgentFramework.reset_memory(settings)  # seeds from packaged example
    assert len(json.load(open(settings.memory_path))) == 2
    data = [opp(f"u{i}").model_dump() for i in range(5)]
    json.dump(data, open(settings.memory_path, "w"))
    DealAgentFramework.reset_memory(settings)
    assert [d["deal"]["url"] for d in json.load(open(settings.memory_path))] == ["u0", "u1"]


def test_seed_memory_is_valid():
    assert all(Opportunity(**d) for d in load_seed_memory())


def test_demo_full_pipeline_offline(settings):
    fw = DealAgentFramework(settings, demo=True)
    memory = fw.run()
    assert isinstance(fw.planner, PlanningAgent)
    # mock pricer yields arbitrary values; the loop must complete and persist if anything beat the threshold
    assert isinstance(memory, list)


def test_build_planner_selects_type(settings, monkeypatch):
    import dataclasses

    from deal_hunter.agents.autonomous_planning import AutonomousPlanningAgent

    class FakeOpenAI:  # avoid needing OPENAI_API_KEY / network
        pass

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    s = dataclasses.replace(settings, specialist_backend="mock", enable_frontier=False, enable_nn=False,
                            enable_preprocessor=False, planner="autonomous")
    assert isinstance(build_planner(s, None), AutonomousPlanningAgent)
    assert isinstance(build_planner(dataclasses.replace(s, planner="workflow"), None), PlanningAgent)
