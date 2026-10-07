"""Glue: audio -> transcript -> minutes, with backend selection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from meeting_minutes.config import Settings, resolve_backend
from meeting_minutes.minutes import ChatAPI, LlamaLocal, MinutesGenerator
from meeting_minutes.transcribe import Transcriber, WhisperAPI, WhisperLocal

_cache: dict[str, object] = {}


def make_transcriber(backend: str) -> Transcriber:
    key = f"t-{backend}"
    if key not in _cache:  # models are heavy: reuse across UI calls
        _cache[key] = WhisperLocal() if backend == "local" else WhisperAPI()
    return _cache[key]  # type: ignore[return-value]


def make_generator(backend: str) -> MinutesGenerator:
    key = f"g-{backend}"
    if key not in _cache:
        _cache[key] = LlamaLocal() if backend == "local" else ChatAPI()
    return _cache[key]  # type: ignore[return-value]


def read_transcript(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def create_minutes(
    audio_path: str | None = None,
    transcript: str | None = None,
    backend: str | None = None,
    meeting_description: str = "a Denver City council meeting",
    transcriber: Transcriber | None = None,
    generator: MinutesGenerator | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(transcript, minutes_so_far)`` pairs.

    Provide ``audio_path`` (transcribed first) or ``transcript`` (skips STT). Backends can be injected for tests.
    """
    if not audio_path and not transcript:
        raise ValueError("Provide an audio file or a transcript.")
    backend = resolve_backend(backend or Settings().backend)
    if not transcript:
        transcriber = transcriber or make_transcriber(backend)
        transcript = transcriber.transcribe(audio_path)  # type: ignore[arg-type]
        yield transcript, ""
    generator = generator or make_generator(backend)
    for minutes in generator.stream(transcript, meeting_description):
        yield transcript, minutes
