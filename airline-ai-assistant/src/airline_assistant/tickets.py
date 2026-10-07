"""In-memory ticket database: prices per destination and bookings."""

from __future__ import annotations

import threading
from dataclasses import dataclass

DEFAULT_PRICES: dict[str, float] = {"london": 799, "paris": 899, "tokyo": 1420, "sydney": 2999}


class TicketError(ValueError):
    """Business-rule violation (unknown city, bad price...). Message is safe to show the LLM."""


@dataclass(frozen=True)
class Booking:
    ticket_id: str
    passenger: str
    city: str
    price: float


def _key(city: str) -> str:
    key = (city or "").strip().lower()
    if not key:
        raise TicketError("A destination city is required")
    return key


def money(value: float) -> str:
    return f"${value:,.0f}" if float(value).is_integer() else f"${value:,.2f}"


class TicketStore:
    """Thread-safe (Gradio serves requests from multiple threads), in-memory only."""

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self._lock = threading.Lock()
        self._prices = {k.lower(): float(v) for k, v in (DEFAULT_PRICES if prices is None else prices).items()}
        self._bookings: list[Booking] = []

    def get_price(self, city: str) -> float | None:
        with self._lock:
            return self._prices.get(_key(city))

    def set_price(self, city: str, price: float) -> None:
        if not isinstance(price, (int, float)) or isinstance(price, bool) or price <= 0:
            raise TicketError("Price must be a positive number")
        with self._lock:
            self._prices[_key(city)] = float(price)

    def book(self, city: str, passenger: str) -> Booking:
        passenger = (passenger or "").strip()
        if not passenger:
            raise TicketError("A passenger name is required to book")
        with self._lock:
            price = self._prices.get(_key(city))
            if price is None:
                raise TicketError(f"We do not fly to {city}")
            booking = Booking(f"FL-{len(self._bookings) + 1:04d}", passenger, _key(city), price)
            self._bookings.append(booking)
            return booking

    def bookings(self) -> list[Booking]:
        with self._lock:
            return list(self._bookings)

    def destinations(self) -> list[str]:
        with self._lock:
            return sorted(self._prices)
