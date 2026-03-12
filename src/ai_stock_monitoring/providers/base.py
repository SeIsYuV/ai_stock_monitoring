from __future__ import annotations

"""Abstract data-provider contracts used by the monitoring engine."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Quote:
    symbol: str
    name: str
    latest_price: float
    updated_at: datetime


@dataclass(frozen=True)
class PriceBar:
    traded_on: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float


class MarketDataProvider:
    """Provider interface so the rest of the app is not tied to one vendor."""
    provider_name = "base"

    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    def get_daily_bars(self, symbol: str, limit: int = 320) -> list[PriceBar]:
        raise NotImplementedError

    def get_weekly_bars(self, symbol: str, limit: int = 80) -> list[PriceBar]:
        raise NotImplementedError

    def get_trailing_dividend_yield(self, symbol: str, latest_price: float) -> float:
        raise NotImplementedError

    def get_trade_dates(self) -> list[date]:
        raise NotImplementedError
