"""Speech-to-text backends.

* ``WhisperLocal``  - HF ``pipeline("automatic-speech-recognition")`` with openai/whisper-medium (GPU recommended).
* ``WhisperAPI``    - OpenAI audio transcription endpoint (no GPU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from openai import OpenAI

from meeting_minutes.config import Settings

OPENAI_AUDIO_LIMIT_MB = 25


class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> str: ...


def _check_file(audio_path: str) -> Path:
    p = Path(audio_path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return p


class WhisperLocal:
    """Lazy-loads the pipeline on first use (heavy imports stay out of API-only installs)."""

    def __init__(self, model: str | None = None, device: str | None = None):
        self.model = model or Settings().whisper_model
        self.device = device
        self._pipe = None

    def _load(self):
        try:
            import torch  # noqa: PLC0415
            from transformers import pipeline  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - depends on env
            raise RuntimeError("Local backend needs: pip install -e '.[local]'") from e
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            dtype=dtype,
            device=device,
            return_timestamps=True,  # required for audio > 30 s (long-form chunked decoding)
        )

    def transcribe(self, audio_path: str) -> str:
        _check_file(audio_path)
        if self._pipe is None:
            self._load()
        return self._pipe(audio_path)["text"].strip()


class WhisperAPI:
    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        self._client = client
        self.model = model or Settings().openai_transcribe_model

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def transcribe(self, audio_path: str) -> str:
        p = _check_file(audio_path)
        size_mb = p.stat().st_size / 1_000_000
        if size_mb > OPENAI_AUDIO_LIMIT_MB:
            raise ValueError(
                f"{p.name} is {size_mb:.0f} MB; the OpenAI API limit is {OPENAI_AUDIO_LIMIT_MB} MB. "
                "Compress (e.g. ffmpeg -i in.wav -ac 1 -b:a 48k out.mp3), split it, or use the local backend."
            )
        with p.open("rb") as f:
            result = self.client.audio.transcriptions.create(model=self.model, file=f, response_format="text")
        return (result if isinstance(result, str) else result.text).strip()
