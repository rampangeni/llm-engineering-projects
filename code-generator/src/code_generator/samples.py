"""Sample Python programs (files live in ./samples)."""

from pathlib import Path

_DIR = Path(__file__).parent / "samples"

SAMPLE_FILES = {
    "pi (Leibniz series, 200M iterations)": "pi.py",
    "lcg + max_subarray_sum (hard)": "max_subarray_sum.py",
}


def load_sample(label: str) -> str:
    return (_DIR / SAMPLE_FILES[label]).read_text(encoding="utf-8")


PI = load_sample("pi (Leibniz series, 200M iterations)")
PYTHON_HARD = load_sample("lcg + max_subarray_sum (hard)")
