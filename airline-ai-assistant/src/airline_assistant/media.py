"""Image generation (city artwork) and text-to-speech via the OpenAI API."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from .config import Settings


class Media:
    def __init__(self, client: Any, settings: Settings) -> None:
        self.client, self.settings = client, settings

    def draw_city(self, city: str) -> Image.Image | None:
        """Pop-art vacation image of ``city``; None if generation fails (never breaks chat)."""
        try:
            resp = self.client.images.generate(
                model=self.settings.image_model,
                prompt=(f"An image representing a vacation in {city}, showing tourist spots and "
                        f"everything unique about {city}, in a vibrant pop-art style"),
                size="1024x1024",
                n=1,
            )
            return Image.open(BytesIO(base64.b64decode(resp.data[0].b64_json)))
        except Exception as exc:
            print(f"Image generation failed: {exc}")
            return None

    def speak(self, text: str) -> bytes | None:
        """mp3 bytes for ``text``; None if TTS fails."""
        try:
            resp = self.client.audio.speech.create(
                model=self.settings.tts_model, voice=self.settings.tts_voice, input=text)
            return resp.content
        except Exception as exc:
            print(f"Text-to-speech failed: {exc}")
            return None
