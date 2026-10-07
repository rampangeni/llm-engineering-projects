"""Abstract base for agents: colour-coded, name-tagged logging."""

from __future__ import annotations

import logging

from .. import log_utils as lu


class Agent:
    RED, GREEN, YELLOW, BLUE = lu.RED, lu.GREEN, lu.YELLOW, lu.BLUE
    MAGENTA, CYAN, WHITE = lu.MAGENTA, lu.CYAN, lu.WHITE
    BG_BLACK, RESET = lu.BG_BLACK, lu.RESET

    name: str = ""
    color: str = lu.WHITE

    def log(self, message: str) -> None:
        logging.info(f"{self.BG_BLACK}{self.color}[{self.name}] {message}{self.RESET}")
