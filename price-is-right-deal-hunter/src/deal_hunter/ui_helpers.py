"""Gradio-free helpers for the UI (unit-testable)."""

from __future__ import annotations

import queue
import time
from typing import Iterator, List

from .models import Opportunity


def table_for(opps: List[Opportunity]) -> List[list]:
    return [
        [o.deal.product_description, f"${o.deal.price:.2f}", f"${o.estimate:.2f}", f"${o.discount:.2f}", o.deal.url]
        for o in opps
    ]


def html_for(log_data: List[str], lines: int = 18) -> str:
    output = "<br>".join(log_data[-lines:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def stream_run(log_data: List[str], log_queue: "queue.Queue", result_queue: "queue.Queue",
               initial_table: List[list], poll: float = 0.1) -> Iterator[tuple]:
    """Yield (log_data, html, table) while a worker thread fills the queues; ends after the final result."""
    from .log_utils import reformat

    final = None
    while True:
        try:
            log_data.append(reformat(log_queue.get_nowait()))
            yield log_data, html_for(log_data), final or initial_table
        except queue.Empty:
            try:
                final = result_queue.get_nowait()
                yield log_data, html_for(log_data), final or initial_table
            except queue.Empty:
                if final is not None:
                    # drain any log lines emitted just before completion
                    while not log_queue.empty():
                        log_data.append(reformat(log_queue.get_nowait()))
                    yield log_data, html_for(log_data), final
                    break
                time.sleep(poll)
