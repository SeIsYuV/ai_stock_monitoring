from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from .base import MarketDataProvider, PriceBar, Quote


class MockMarketDataProvider(MarketDataProvider):
    provider_name = "mock"

    def get_quote(self, symbol: str) -> Quote:
        digits = [int(char) for char in symbol if char.isdigit()]
        seed = sum(digits)
        base_price = 10 + (seed % 90)
        minute_factor = datetime.now(ZoneInfo("Asia/Shanghai")).minute % 6
        latest_price = round(base_price + minute_factor * 0.17, 2)
        return Quote(
            symbol=symbol,
            name=f"示例股票{symbol}",
            latest_price=latest_price,
            updated_at=datetime.now(ZoneInfo("Asia/Shanghai")),
        )

    def get_daily_bars(self, symbol: str, limit: int = 320) -> list[PriceBar]:
        base = self.get_quote(symbol).latest_price
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        return [
            PriceBar(
                traded_on=today - timedelta(days=offset),
                open_price=round(base * 0.99, 2),
                high_price=round(base * 1.01, 2),
                low_price=round(base * 0.98, 2),
                close_price=round(base + (offset % 7 - 3) * 0.12, 2),
            )
            for offset in reversed(range(limit))
        ]

    def get_weekly_bars(self, symbol: str, limit: int = 80) -> list[PriceBar]:
        base = self.get_quote(symbol).latest_price
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        return [
            PriceBar(
                traded_on=today - timedelta(days=offset * 7),
                open_price=round(base * 0.98, 2),
                high_price=round(base * 1.02, 2),
                low_price=round(base * 0.96, 2),
                close_price=round(base + (offset % 5 - 2) * 0.35, 2),
            )
            for offset in reversed(range(limit))
        ]

    def get_trailing_dividend_yield(self, symbol: str, latest_price: float) -> float:
        seed = sum(int(char) for char in symbol if char.isdigit())
        if latest_price <= 0:
            return 0.0
        last_year_dividend_per_share = round(0.18 + (seed % 8) * 0.06, 2)
        return round(last_year_dividend_per_share / latest_price * 100, 2)

    def get_trade_dates(self) -> list[date]:
        today = datetime.now(UTC).date()
        start = today - timedelta(days=400)
        dates: list[date] = []
        current = start
        while current <= today + timedelta(days=400):
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)
        return dates
