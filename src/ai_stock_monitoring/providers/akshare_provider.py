from __future__ import annotations

"""AKShare-based market data provider.

职责边界：
- 只负责向外部数据源取数
- 不负责提醒逻辑
- 不负责数据库写入

这样后面如果想替换成别的数据源，只需要保持这里的接口一致。
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable

import akshare as ak
import pandas as pd

from .base import MarketDataProvider, PriceBar, Quote


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


class AkshareMarketDataProvider(MarketDataProvider):
    """Fetch quotes, history bars, dividends and trade dates through AKShare."""
    provider_name = "akshare"

    def __init__(self) -> None:
        # 轻量内存缓存：减少同一轮页面刷新里反复打外部接口。
        self._cache: dict[tuple[str, str], tuple[datetime, Any]] = {}

    def get_quote(self, symbol: str) -> Quote:
        """Load the latest quote and display name for one symbol."""
        item_map = self._load_symbol_item_map(symbol)
        latest_price = _to_float(item_map.get("最新") or item_map.get("最新价"))
        name = str(item_map.get("股票简称") or symbol)
        return Quote(
            symbol=symbol,
            name=name,
            latest_price=latest_price,
            updated_at=datetime.now(UTC),
        )

    def get_daily_bars(self, symbol: str, limit: int = 320) -> list[PriceBar]:
        """Load daily forward-adjusted bars, enough for 250-day calculations."""
        start_date = (datetime.now(UTC).date() - timedelta(days=800)).strftime("%Y%m%d")
        frame = self._cached(
            key=("daily", symbol),
            ttl_seconds=45,
            builder=lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date="20500101",
                adjust="qfq",
            ),
        )
        return self._to_bars(frame.tail(limit))

    def get_weekly_bars(self, symbol: str, limit: int = 80) -> list[PriceBar]:
        """Load weekly forward-adjusted bars for 30/60-week MA calculations."""
        start_date = (datetime.now(UTC).date() - timedelta(days=900)).strftime("%Y%m%d")
        frame = self._cached(
            key=("weekly", symbol),
            ttl_seconds=900,
            builder=lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="weekly",
                start_date=start_date,
                end_date="20500101",
                adjust="qfq",
            ),
        )
        return self._to_bars(frame.tail(limit))

    def get_trailing_dividend_yield(self, symbol: str, latest_price: float) -> float:
        """Compute dividend yield using last calendar year's total cash dividend per share / current price."""
        if latest_price <= 0:
            return 0.0
        detail = self._cached(
            key=("dividend", symbol),
            ttl_seconds=21600,
            builder=lambda: ak.stock_history_dividend_detail(symbol=symbol, indicator="分红"),
        )
        if detail.empty:
            return 0.0

        target_year = datetime.now(UTC).date().year - 1
        working_frame = detail.copy()
        working_frame["除权除息日"] = pd.to_datetime(working_frame["除权除息日"], errors="coerce")
        working_frame["公告日期"] = pd.to_datetime(working_frame["公告日期"], errors="coerce")
        working_frame["派息"] = pd.to_numeric(working_frame["派息"], errors="coerce").fillna(0.0)
        effective_dates = working_frame["除权除息日"].fillna(working_frame["公告日期"])
        filtered = working_frame.loc[
            (effective_dates.dt.year == target_year)
            & (working_frame["进度"] == "实施")
        ]
        total_cash_per_10 = float(filtered["派息"].sum())
        if total_cash_per_10 <= 0:
            return 0.0
        return round(total_cash_per_10 / 10 / latest_price * 100, 2)

    def get_trade_dates(self) -> list[date]:
        frame = self._cached(
            key=("calendar", "cn"),
            ttl_seconds=21600,
            builder=ak.tool_trade_date_hist_sina,
        )
        trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        return [item.date() for item in trade_dates.to_list()]


    def get_reference_index_daily_bars(self, index_symbol: str, limit: int = 120) -> list[PriceBar]:
        start_date = (datetime.now(UTC).date() - timedelta(days=260)).strftime("%Y%m%d")
        try:
            frame = self._cached(
                key=("index_daily", index_symbol),
                ttl_seconds=300,
                builder=lambda: ak.stock_zh_index_daily_em(symbol=index_symbol),
            )
        except Exception:
            return []
        if frame is None or frame.empty:
            return []
        working = frame.copy().tail(limit)
        renamed = working.rename(columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
        })
        if "日期" not in renamed.columns:
            return []
        return self._to_bars(renamed.tail(limit))

    def get_symbol_profile(self, symbol: str) -> dict[str, Any]:
        item_map = self._load_symbol_item_map(symbol)
        industry_name = str(item_map.get("行业") or item_map.get("所属行业") or "")
        return {"industry_name": industry_name}

    def get_industry_daily_bars(self, industry_name: str, limit: int = 120) -> list[PriceBar]:
        if not industry_name:
            return []
        start_date = (datetime.now(UTC).date() - timedelta(days=260)).strftime("%Y%m%d")
        try:
            frame = self._cached(
                key=("industry_daily", industry_name),
                ttl_seconds=1800,
                builder=lambda: ak.stock_board_industry_hist_em(
                    symbol=industry_name,
                    start_date=start_date,
                    end_date="20500101",
                    period="日k",
                    adjust="",
                ),
            )
        except Exception:
            return []
        if frame is None or frame.empty:
            return []
        return self._to_bars(frame.tail(limit))

    def _load_symbol_item_map(self, symbol: str) -> dict[str, Any]:
        info = self._cached(
            key=("quote", symbol),
            ttl_seconds=20,
            builder=lambda: ak.stock_individual_info_em(symbol=symbol),
        )
        return {
            str(row["item"]): row["value"]
            for _, row in info.iterrows()
        }

    def _cached(
        self,
        key: tuple[str, str],
        ttl_seconds: int,
        builder: Callable[[], Any],
    ) -> Any:
        """Return cached provider data when TTL has not expired yet."""

        now = datetime.now(UTC)
        cached_item = self._cache.get(key)
        if cached_item and cached_item[0] > now:
            return cached_item[1]
        value = builder()
        self._cache[key] = (now + timedelta(seconds=ttl_seconds), value)
        return value

    @staticmethod
    def _to_bars(frame: pd.DataFrame) -> list[PriceBar]:
        bars: list[PriceBar] = []
        for _, row in frame.iterrows():
            traded_on = pd.to_datetime(row["日期"], errors="coerce")
            if pd.isna(traded_on):
                continue
            bars.append(
                PriceBar(
                    traded_on=traded_on.date(),
                    open_price=_to_float(row["开盘"]),
                    high_price=_to_float(row["最高"]),
                    low_price=_to_float(row["最低"]),
                    close_price=_to_float(row["收盘"]),
                    volume=_to_float(row.get("成交量", 0.0)),
                )
            )
        return bars
