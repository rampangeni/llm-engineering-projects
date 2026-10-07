"""Run Python, compile + run C++/Rust.

SECURITY: all of this executes code that was either typed by you or *generated
by an LLM*. There is no real sandbox. We reduce blast radius with a temp working
directory, a timeout, a scrubbed environment (API keys are NOT passed to
children) and no shell, but the code still runs with your user's privileges.
Run inside a container/VM if the inputs are not trusted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from code_generator.prompts import LANGUAGES
from code_generator.system_info import pick_cpp_compile_command, pick_rust_compile_command

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def run_timeout() -> float:
    return float(os.getenv("RUN_TIMEOUT", "60"))


def compile_timeout() -> float:
    return float(os.getenv("COMPILE_TIMEOUT", "120"))


def scrubbed_env() -> dict[str, str]:
    """Environment for child processes without anything that looks like a secret."""
    return {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_MARKERS)}


@dataclass
class RunResult:
    ok: bool
    output: str  # stdout, or error text
    seconds: float  # wall time of the run step (not compile)

    def render(self) -> str:
        return self.output if self.ok else f"An error occurred:\n{self.output}"


def _run(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=scrubbed_env(), text=True, capture_output=True, timeout=timeout, check=False
    )


def run_python(code: str, timeout: float | None = None) -> RunResult:
    """Run Python in a separate interpreter (not in-process exec)."""
    timeout = timeout or run_timeout()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "main.py"
        path.write_text(code, encoding="utf-8")
        start = time.perf_counter()
        try:
            p = _run([sys.executable, str(path)], tmp, timeout)
        except subprocess.TimeoutExpired:
            return RunResult(False, f"Timed out after {timeout:.0f}s", timeout)
        secs = time.perf_counter() - start
    if p.returncode != 0:
        return RunResult(False, (p.stderr or p.stdout).strip(), secs)
    return RunResult(True, p.stdout, secs)


def compile_command_for(language: str, source: str, output: str) -> list[str]:
    if language == "C++":
        return pick_cpp_compile_command(source=source, output=output)
    if language == "Rust":
        return pick_rust_compile_command(source=source, output=output)
    raise ValueError(f"Unsupported language: {language}")


def compile_and_run(
    code: str,
    language: str,
    compile_command: list[str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Write ``code`` to a temp dir, compile it, run the binary.

    ``compile_command`` (if given) must reference ``main.<ext>`` and ``main``.
    """
    meta = LANGUAGES[language]
    timeout = timeout or run_timeout()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / meta["filename"]).write_text(code, encoding="utf-8")
        try:
            cmd = compile_command or compile_command_for(language, meta["filename"], "main")
        except RuntimeError as e:
            return RunResult(False, str(e), 0.0)
        try:
            c = _run(cmd, tmp, compile_timeout())
        except FileNotFoundError as e:
            return RunResult(False, f"Compiler not found: {e}", 0.0)
        except subprocess.TimeoutExpired:
            return RunResult(False, "Compilation timed out", 0.0)
        if c.returncode != 0:
            return RunResult(False, c.stderr.strip() or c.stdout.strip(), 0.0)
        binary = str(Path(tmp) / ("main.exe" if os.name == "nt" else "main"))
        start = time.perf_counter()
        try:
            p = _run([binary], tmp, timeout)
        except subprocess.TimeoutExpired:
            return RunResult(False, f"Timed out after {timeout:.0f}s", timeout)
        secs = time.perf_counter() - start
    if p.returncode != 0:
        return RunResult(False, (p.stderr or p.stdout).strip(), secs)
    return RunResult(True, p.stdout, secs)


def speedup(python_seconds: float, native_seconds: float) -> float | None:
    """python_time / native_time."""
    if native_seconds <= 0:
        return None
    return python_seconds / native_seconds
