import logging
import queue
import threading

from conftest import make_deal
from deal_hunter import cli
from deal_hunter.log_utils import (BG_BLACK, BG_BLUE, GREEN, RESET, WHITE, QueueHandler, attach_queue_handler,
                                   detach_handler, reformat)
from deal_hunter.models import Opportunity
from deal_hunter.ui_helpers import html_for, stream_run, table_for


def test_reformat_maps_agent_colours_to_html():
    out = reformat(f"{BG_BLACK}{GREEN}[Planning Agent] hi{RESET}")
    assert out == '<span style="color: #00dd00">[Planning Agent] hi</span>'
    assert "#ff7800" in reformat(f"{BG_BLUE}{WHITE}[Agent Framework] x{RESET}")


def test_queue_handler_receives_agent_logs():
    from deal_hunter.agents.base import Agent

    q = queue.Queue()
    h = attach_queue_handler(q)
    try:
        class A(Agent):
            name, color = "Test Agent", Agent.CYAN

        A().log("hello")
    finally:
        detach_handler(h)
    assert "[Test Agent] hello" in q.get_nowait()
    logging.info("after detach")
    assert q.empty()


def test_table_and_html_helpers():
    o = Opportunity(deal=make_deal(price=12.5), estimate=40, discount=27.5)
    assert table_for([o])[0][1:4] == ["$12.50", "$40.00", "$27.50"]
    html = html_for([str(i) for i in range(100)])
    assert "99" in html and ">5<" not in html and "scrollContent" in html


def test_stream_run_yields_logs_then_final():
    lq, rq = queue.Queue(), queue.Queue()

    def worker():
        lq.put(f"{BG_BLACK}{GREEN}step{RESET}")
        rq.put([["row"]])

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    outs = list(stream_run([], lq, rq, [["initial"]], poll=0.001))
    assert outs[-1][2] == [["row"]]
    assert any("step" in h for h in outs[-1][0])


def test_cli_parser_and_check(capsys, monkeypatch, tmp_path):
    p = cli.build_parser()
    assert p.parse_args(["agent", "ensemble", "some text"]).name == "ensemble"
    assert p.parse_args(["build-vectorstore", "--lite", "--limit", "10"]).limit == 10
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPECIALIST_BACKEND", "mock")
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "NN weights" in out and "MISSING" in out


def test_cli_run_once_demo(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run-once", "--demo"]) == 0
    assert capsys.readouterr().out.strip().startswith("[")


def test_cli_agent_scan_demo_and_reset_memory(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["agent", "scan", "--demo"]) == 0
    assert "Hisense" in capsys.readouterr().out
    assert cli.main(["reset-memory"]) == 0
    assert (tmp_path / "memory.json").exists()
