"""Colour-coded logging. Agents log ANSI colours to the terminal; the Gradio UI converts the
same strings into HTML spans via ``reformat``."""

from __future__ import annotations

import logging
import sys

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

mapper = {
    BG_BLACK + RED: "#dd0000",
    BG_BLACK + GREEN: "#00dd00",
    BG_BLACK + YELLOW: "#dddd00",
    BG_BLACK + BLUE: "#0000ee",
    BG_BLACK + MAGENTA: "#aa00dd",
    BG_BLACK + CYAN: "#00dddd",
    BG_BLACK + WHITE: "#87CEEB",
    BG_BLUE + WHITE: "#ff7800",
}


def reformat(message: str) -> str:
    """ANSI colour codes -> HTML spans."""
    for key, value in mapper.items():
        message = message.replace(key, f'<span style="color: {value}">')
    return message.replace(RESET, "</span>")


_STREAM_MARK = "_deal_hunter_stream"


def init_logging(level: int = logging.INFO) -> None:
    """Idempotently attach a stdout handler to the root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(h, _STREAM_MARK, False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [Agents] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z")
    )
    setattr(handler, _STREAM_MARK, True)
    root.addHandler(handler)


class QueueHandler(logging.Handler):
    """Pushes formatted log lines onto a queue so the UI can stream them."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def attach_queue_handler(log_queue) -> QueueHandler:
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler


def detach_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
