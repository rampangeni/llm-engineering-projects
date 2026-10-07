"""Wire settings, client, ticket store, tools and media into an Assistant."""

from __future__ import annotations

from .assistant import Assistant
from .config import Settings, load_settings, make_client
from .media import Media
from .tickets import TicketStore
from .tools import build_default_registry


def build_assistant(settings: Settings | None = None, media: bool = True,
                    store: TicketStore | None = None) -> Assistant:
    settings = settings or load_settings()
    client = make_client(settings)
    registry = build_default_registry(store or TicketStore(), include_admin=settings.admin_tools)
    use_media = media and settings.media_supported
    return Assistant(client, settings, registry, Media(client, settings) if use_media else None)
