from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from .base import MarketDataProvider, PriceBar, Quote


class MockMarketDataProvider(MarketDataProvider):
    provider_name = "mock"
    _INDUSTRIES = ("白酒", "银行", "新能源", "医药", "消费电子")

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
                volume=round(8_000_000 + ((offset + sum(int(char) for char in symbol if char.isdigit())) % 11) * 650_000, 2),
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
                volume=round(42_000_000 + ((offset + sum(int(char) for char in symbol if char.isdigit())) % 9) * 1_200_000, 2),
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

    def get_reference_index_daily_bars(self, index_symbol: str, limit: int = 120) -> list[PriceBar]:
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        seed = sum(ord(char) for char in index_symbol)
        base = 3200 + (seed % 600)
        slope = ((seed % 5) - 1) * 4.5
        return [
            PriceBar(
                traded_on=today - timedelta(days=offset),
                open_price=round(base + slope * (limit - offset - 1) - 12, 2),
                high_price=round(base + slope * (limit - offset - 1) + 24, 2),
                low_price=round(base + slope * (limit - offset - 1) - 28, 2),
                close_price=round(base + slope * (limit - offset - 1) + ((offset % 6) - 2) * 3.2, 2),
                volume=round(180_000_000 + ((offset + seed) % 13) * 8_500_000, 2),
            )
            for offset in reversed(range(limit))
        ]

    def get_symbol_profile(self, symbol: str) -> dict[str, str]:
        seed = sum(int(char) for char in symbol if char.isdigit())
        return {"industry_name": self._INDUSTRIES[seed % len(self._INDUSTRIES)]}

    def get_industry_daily_bars(self, industry_name: str, limit: int = 120) -> list[PriceBar]:
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        seed = sum(ord(char) for char in industry_name)
        base = 980 + (seed % 220)
        slope = ((seed % 7) - 2) * 1.6
        return [
            PriceBar(
                traded_on=today - timedelta(days=offset),
                open_price=round(base + slope * (limit - offset - 1) - 4, 2),
                high_price=round(base + slope * (limit - offset - 1) + 8, 2),
                low_price=round(base + slope * (limit - offset - 1) - 10, 2),
                close_price=round(base + slope * (limit - offset - 1) + ((offset % 5) - 2) * 1.4, 2),
                volume=round(55_000_000 + ((offset + seed) % 10) * 2_600_000, 2),
            )
            for offset in reversed(range(limit))
        ]
