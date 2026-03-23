from __future__ import annotations

"""Trading calendar and market-hour helpers.

这个模块只负责“时间判断”，不关心行情、邮件或数据库。
把时间逻辑单独抽出来后，测试会简单很多。
"""

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketStatus:
    label: str
    is_trading_day: bool
    is_market_open: bool
    is_pre_open_window: bool
    is_post_close: bool
    next_refresh_seconds: int | None
    current_time: datetime


class TradeCalendar:
    """Small wrapper around a list of trade dates.

    如果实时交易日历加载失败，会退回到“工作日”模式，
    这样应用至少还能启动和演示。
    """
    def __init__(self, trade_dates: list[date] | None = None) -> None:
        self.trade_dates = sorted(trade_dates or [])
        self.trade_date_set = set(self.trade_dates)

    def is_trading_day(self, current_date: date) -> bool:
        if self.trade_date_set:
            return current_date in self.trade_date_set
        return current_date.weekday() < 5

    def next_trading_day(self, current_date: date) -> date | None:
        for item in self.trade_dates:
            if item > current_date:
                return item
        return None

    def is_last_trading_day_of_week(self, current_date: date) -> bool:
        next_day = self.next_trading_day(current_date)
        if next_day is None:
            return True
        return next_day.isocalendar()[:2] != current_date.isocalendar()[:2]

    def is_last_trading_day_of_month(self, current_date: date) -> bool:
        next_day = self.next_trading_day(current_date)
        if next_day is None:
            return True
        return (next_day.year, next_day.month) != (current_date.year, current_date.month)

    def is_last_trading_day_of_year(self, current_date: date) -> bool:
        next_day = self.next_trading_day(current_date)
        if next_day is None:
            return True
        return next_day.year != current_date.year


def get_market_status(
    refresh_interval_seconds: int,
    trade_calendar: TradeCalendar | None = None,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> MarketStatus:
    """Translate the current time into the app-specific market status."""

    current_time = now.astimezone(ZoneInfo(timezone_name)) if now else datetime.now(
        ZoneInfo(timezone_name)
    )
    current_date = current_time.date()
    calendar = trade_calendar or TradeCalendar()
    is_trading_day = calendar.is_trading_day(current_date)
    pre_open_start = time(hour=9, minute=25)
    market_open = time(hour=9, minute=30)
    market_close = time(hour=15, minute=0)
    current_clock = current_time.time()

    is_pre_open_window = is_trading_day and pre_open_start <= current_clock < market_open
    is_market_open = is_trading_day and market_open <= current_clock < market_close
    is_post_close = is_trading_day and current_clock >= market_close

    if is_market_open:
        seconds_since_open = int(
            (current_time - current_time.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds()
        )
        next_refresh_seconds = refresh_interval_seconds - (
            seconds_since_open % refresh_interval_seconds
        )
        label = "监控中（A股 9:30-15:00）"
    elif is_pre_open_window:
        next_refresh_seconds = int(
            (
                datetime.combine(current_date, market_open, tzinfo=current_time.tzinfo)
                - current_time
            ).total_seconds()
        )
        label = "开盘准备中（9:25-9:30）"
    elif not is_trading_day:
        next_refresh_seconds = None
        label = "非交易日，监控暂停"
    elif current_clock < pre_open_start:
        next_refresh_seconds = int(
            (
                datetime.combine(current_date, pre_open_start, tzinfo=current_time.tzinfo)
                - current_time
            ).total_seconds()
        )
        label = "监控未启动（等待开盘）"
    else:
        next_refresh_seconds = None
        label = "监控已停止（非开盘时段）"

    return MarketStatus(
        label=label,
        is_trading_day=is_trading_day,
        is_market_open=is_market_open,
        is_pre_open_window=is_pre_open_window,
        is_post_close=is_post_close,
        next_refresh_seconds=next_refresh_seconds,
        current_time=current_time,
    )
