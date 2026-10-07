"""FlightAI: tool-calling multimodal airline assistant."""

from .assistant import Assistant, AssistantReply
from .tickets import TicketStore
from .tools import ToolRegistry, build_default_registry

__all__ = ["Assistant", "AssistantReply", "TicketStore", "ToolRegistry", "build_default_registry"]
__version__ = "0.1.0"
