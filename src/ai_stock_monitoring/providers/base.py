from __future__ import annotations

"""Abstract data-provider contracts used by the monitoring engine."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


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
    volume: float = 0.0


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

    def get_reference_index_daily_bars(self, index_symbol: str, limit: int = 120) -> list[PriceBar]:
        return []

    def get_symbol_profile(self, symbol: str) -> dict[str, Any]:
        return {"industry_name": ""}

    def get_industry_daily_bars(self, industry_name: str, limit: int = 120) -> list[PriceBar]:
        return []

    def get_symbol_fundamentals(self, symbol: str) -> dict[str, float | None]:
        return {
            "pe_ttm": None,
            "pb": None,
            "market_cap": None,
        }

    def invalidate_symbol_cache(self, symbol: str) -> None:
        """Drop short-lived cache entries for one symbol when a hard refresh is requested."""

    def invalidate_all_cache(self) -> None:
        """Drop all short-lived cache entries when the caller needs a full refresh."""
