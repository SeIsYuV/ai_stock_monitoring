from __future__ import annotations

"""Monitoring engine for market checks, alerts and chart data.

这是业务核心模块。
如果你只想理解“系统到底怎么判断要不要提醒”，优先看：
- `run_cycle`：整轮监控的调度入口
- `_refresh_open_session`：盘中刷新逻辑
- `_prepare_dividend_alerts`：开盘前股息率提醒
- `_prepare_weekly_cross_alerts`：周线交叉次日提醒
- `_process_intraday_signal`：按不同信号类型决定是否真正触发告警
- `build_snapshot`：统一计算技术指标与量化综合评分
"""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
import json
import math
import re
import threading
from typing import Any

SELL_DIVIDEND_THRESHOLD = 3.5
QUANT_SELL_PROBABILITY_THRESHOLD = 35.0
BUY_ALERT_MIN_LEVEL = 7
BUY_ALERT_MIN_QUANT_PROBABILITY = 60.0
BUY_ALERT_STRONG_QUANT_PROBABILITY = 80.0
BUY_ALERT_DIVIDEND_CONFIRMATION_QUANT = 78.0
BUY_ALERT_WEEKLY_CONFIRMATION_QUANT = 70.0
BUY_ALERT_MIN_LEVEL_GAP = 2
BUY_ALERT_MAX_MODEL_DISPERSION = 18.0
BUY_ALERT_MIN_EXPECTED_UPSIDE_PCT = 4.0
BUY_ALERT_MIN_DCF_GAP_PCT = -5.0
SELL_ALERT_MIN_LEVEL = 5
SELL_ALERT_STRONG_LEVEL = 7
SELL_ALERT_BOLL_EXTENSION_RATIO = 1.02
SELL_ALERT_LOW_DIVIDEND_MAX_QUANT = 55.0
WEEKLY_SUPPORT_TRIGGER_TYPES = {"30周/60周均线", "30周线上穿60周线", "有效突破60周线"}
BUY_TRIGGER_TYPES = {"250日线", "BOLL中轨", "BOLL下轨", "股息率", "30周/60周均线", "30周线上穿60周线", "有效突破60周线", "量化盈利概率"}
PRICE_SUPPORT_TRIGGER_TYPES = {"250日线", "BOLL中轨", "BOLL下轨"}
SELL_TRIGGER_TYPES = {"BOLL上轨卖出", "低股息率卖出", "量化走弱卖出"}
WEEKLY_BREAKOUT_MIN_CLOSE_RATIO = 1.01
WEEKLY_BREAKOUT_MIN_LOW_RATIO = 0.995
PAPER_TRADE_OPEN_THRESHOLD_MODEL = 72.0
PAPER_TRADE_CLOSE_THRESHOLD_MODEL = 46.0
PAPER_TRADE_OPEN_THRESHOLD_GROUP = 68.0
PAPER_TRADE_CLOSE_THRESHOLD_GROUP = 50.0
PAPER_TRADE_TAKE_PROFIT_PCT = 12.0
PAPER_TRADE_STOP_LOSS_PCT = -8.0
PAPER_TRADE_MAX_HOLDING_DAYS_MODEL = 12
PAPER_TRADE_MAX_HOLDING_DAYS_GROUP = 20

from .config import AppSettings
from .database import (
    add_alert_history,
    close_model_paper_trade,
    get_email_settings,
    get_job_state,
    get_monitored_stocks,
    get_open_model_paper_trade,
    get_portfolio_settings,
    get_quant_settings,
    get_snapshot,
    get_signal_state,
    list_model_paper_trades,
    list_pending_signal_states,
    list_trade_records,
    list_trade_records_for_symbol,
    mark_model_paper_trade,
    open_model_paper_trade,
    set_job_state,
    upsert_signal_state,
    upsert_snapshot,
)
from .mailer import (
    build_alert_email_body,
    build_alert_email_html_body,
    build_portfolio_review_email_body,
    build_portfolio_review_email_html_body,
    send_message,
)
from .market_hours import MarketStatus, TradeCalendar, get_market_status
from .providers import load_provider
from .providers.base import PriceBar, Quote
from .quant import (
    DEFAULT_QUANT_MODELS,
    PROFESSIONAL_MODEL_KEYS,
    build_quant_signal,
    normalize_selected_models,
    normalize_strategy_params,
)
from .trade_advisor import build_portfolio_profile, build_position_summary, build_stock_comprehensive_advice


@dataclass(frozen=True)
class QuantSellSignal:
    should_alert: bool
    confirmation_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotComputation:
    symbol: str
    display_name: str
    latest_price: float
    ma_250: float
    ma_30w: float
    ma_60w: float
    prev_ma_30w: float
    prev_ma_60w: float
    boll_mid: float
    boll_lower: float
    boll_upper: float
    dividend_yield: float
    quant_probability: float
    quant_model_breakdown: str
    trigger_state: str
    trigger_detail: str
    triggered_labels: tuple[str, ...]
    weekly_crossed: bool
    updated_at: datetime
    latest_change_amount: float = 0.0
    latest_change_pct: float = 0.0
    latest_volume_ratio: float = 1.0
    market_environment: str = "中性"
    market_bias_score: float = 0.0
    industry_name: str = ""
    industry_environment: str = "中性"
    industry_bias_score: float = 0.0
    earnings_phase: str = "常规窗口"
    earnings_days_to_window: int = 999
    position_concentration_pct: float = 0.0
    weekly_bullish_crossed: bool = False
    weekly_bearish_crossed: bool = False
    weekly_close: float = 0.0
    prev_weekly_close: float = 0.0
    weekly_breakout_above_ma60w: bool = False


def validate_stock_symbol(symbol: str) -> bool:
    return len(symbol) == 6 and symbol.isdigit()


def parse_stock_symbols(raw_text: str) -> tuple[list[str], list[str]]:
    candidates = [item.strip() for item in re.split(r"[\s,，;；]+", raw_text) if item.strip()]
    valid_symbols: list[str] = []
    invalid_symbols: list[str] = []
    for symbol in candidates:
        if validate_stock_symbol(symbol):
            if symbol not in valid_symbols:
                valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)
    return valid_symbols, invalid_symbols


def calculate_simple_moving_average(values: list[float], window: int) -> float:
    if len(values) < window:
        return 0.0
    return round(sum(values[-window:]) / window, 2)


def calculate_bollinger_lower_band(values: list[float], window: int = 20, std_multiplier: float = 2.0) -> float:
    """Compute a simple BOLL lower band from the latest closing prices."""

    if len(values) < window:
        return 0.0
    sample = values[-window:]
    middle = sum(sample) / window
    variance = sum((item - middle) ** 2 for item in sample) / window
    lower_band = middle - std_multiplier * math.sqrt(variance)
    return round(lower_band, 2)


def calculate_bollinger_upper_band(values: list[float], window: int = 20, std_multiplier: float = 2.0) -> float:
    """Compute a simple BOLL upper band from the latest closing prices."""

    if len(values) < window:
        return 0.0
    sample = values[-window:]
    middle = sum(sample) / window
    variance = sum((item - middle) ** 2 for item in sample) / window
    upper_band = middle + std_multiplier * math.sqrt(variance)
    return round(upper_band, 2)




def calculate_volume_ratio(daily_bars: list[PriceBar], short_window: int = 1, long_window: int = 20) -> float:
    if len(daily_bars) < long_window:
        return 1.0
    recent_volumes = [float(item.volume or 0.0) for item in daily_bars[-short_window:]]
    baseline_volumes = [float(item.volume or 0.0) for item in daily_bars[-long_window:]]
    baseline = sum(baseline_volumes) / max(len(baseline_volumes), 1)
    if baseline <= 0:
        return 1.0
    recent = sum(recent_volumes) / max(len(recent_volumes), 1)
    return round(recent / baseline, 2)


def classify_trend_environment(bars: list[PriceBar]) -> tuple[str, float]:
    closes = [float(item.close_price) for item in bars if float(item.close_price or 0.0) > 0]
    if len(closes) < 60:
        return "中性", 0.0
    latest_close = closes[-1]
    ma_20 = calculate_simple_moving_average(closes, 20)
    ma_60 = calculate_simple_moving_average(closes, 60)
    score = 0.0
    if latest_close > ma_20 > 0:
        score += 18.0
    elif latest_close < ma_20:
        score -= 18.0
    if ma_20 > ma_60 > 0:
        score += 16.0
    elif 0 < ma_20 < ma_60:
        score -= 16.0
    ten_day_base = closes[-11] if len(closes) >= 11 else closes[0]
    if ten_day_base > 0:
        ten_day_return_pct = (latest_close - ten_day_base) / ten_day_base * 100
        score += max(-18.0, min(18.0, ten_day_return_pct * 2.0))
    if latest_close > 0:
        distance_pct = (latest_close - ma_60) / ma_60 * 100 if ma_60 > 0 else 0.0
        score += max(-12.0, min(12.0, distance_pct / 2.0))
    if score >= 18.0:
        return "偏强", round(score, 2)
    if score <= -18.0:
        return "偏弱", round(score, 2)
    return "中性", round(score, 2)


def infer_earnings_phase(trade_date: date) -> tuple[str, int]:
    candidate_days: list[int] = []
    for year in (trade_date.year - 1, trade_date.year, trade_date.year + 1):
        for month, day in ((4, 30), (8, 31), (10, 31)):
            candidate_days.append(abs((date(year, month, day) - trade_date).days))
    nearest_days = min(candidate_days) if candidate_days else 999
    if nearest_days <= 7:
        return "财报窗口进行中", nearest_days
    if nearest_days <= 21:
        return "财报窗口临近", nearest_days
    return "常规窗口", nearest_days

def build_quant_sell_signal(snapshot: SnapshotComputation) -> QuantSellSignal:
    """Use the current snapshot to decide whether a sell-side quant alert should fire."""

    confirmation_reasons: list[str] = []
    if snapshot.boll_mid and snapshot.latest_price < snapshot.boll_mid:
        confirmation_reasons.append("价格跌回 BOLL 中轨下方")
    if snapshot.ma_250 and snapshot.latest_price < snapshot.ma_250:
        confirmation_reasons.append("价格跌回 250 日线下方")
    if snapshot.ma_30w and snapshot.ma_60w and snapshot.ma_30w < snapshot.ma_60w:
        confirmation_reasons.append("30 周均线弱于 60 周均线")

    should_alert = snapshot.quant_probability <= QUANT_SELL_PROBABILITY_THRESHOLD and len(confirmation_reasons) >= 2
    reasons: list[str] = [f"量化综合盈利概率降至 {snapshot.quant_probability:.2f}%"]
    reasons.extend(confirmation_reasons)
    return QuantSellSignal(
        should_alert=should_alert,
        confirmation_count=len(confirmation_reasons),
        reasons=tuple(reasons),
    )


def has_weekly_crossed(
    prev_ma_30w: float,
    prev_ma_60w: float,
    current_ma_30w: float,
    current_ma_60w: float,
) -> bool:
    return get_weekly_cross_direction(prev_ma_30w, prev_ma_60w, current_ma_30w, current_ma_60w) is not None


def get_weekly_cross_direction(
    prev_ma_30w: float,
    prev_ma_60w: float,
    current_ma_30w: float,
    current_ma_60w: float,
) -> str | None:
    if not all([prev_ma_30w, prev_ma_60w, current_ma_30w, current_ma_60w]):
        return None
    prev_diff = prev_ma_30w - prev_ma_60w
    current_diff = current_ma_30w - current_ma_60w
    if (prev_diff <= 0 < current_diff) or (prev_diff < 0 <= current_diff):
        return "bullish"
    if (prev_diff >= 0 > current_diff) or (prev_diff > 0 >= current_diff):
        return "bearish"
    return None


def has_effective_breakout_above_ma60w(
    weekly_bars: list[PriceBar],
    current_ma_60w: float,
    prev_ma_60w: float,
    min_close_ratio: float = WEEKLY_BREAKOUT_MIN_CLOSE_RATIO,
    min_low_ratio: float = WEEKLY_BREAKOUT_MIN_LOW_RATIO,
) -> bool:
    if len(weekly_bars) < 2 or not current_ma_60w or not prev_ma_60w:
        return False
    latest_week = weekly_bars[-1]
    previous_week = weekly_bars[-2]
    if previous_week.close_price > prev_ma_60w:
        return False
    if latest_week.close_price < current_ma_60w * min_close_ratio:
        return False
    if latest_week.low_price < current_ma_60w * min_low_ratio:
        return False
    return True


def compute_snapshot_metrics(
    symbol: str,
    quote: Quote,
    daily_bars: list[PriceBar],
    weekly_bars: list[PriceBar],
    dividend_yield: float,
    quant_probability: float,
    quant_model_breakdown: str,
) -> SnapshotComputation:
    daily_closes = [item.close_price for item in daily_bars]
    weekly_closes = [item.close_price for item in weekly_bars]
    previous_close = daily_closes[-2] if len(daily_closes) >= 2 else (daily_closes[-1] if daily_closes else quote.latest_price)
    latest_change_amount = round(quote.latest_price - previous_close, 2)
    latest_change_pct = round((latest_change_amount / previous_close) * 100, 2) if previous_close else 0.0

    ma_250 = calculate_simple_moving_average(daily_closes, 250)
    boll_mid = calculate_simple_moving_average(weekly_closes, 20)
    boll_lower = calculate_bollinger_lower_band(weekly_closes, 20)
    boll_upper = calculate_bollinger_upper_band(weekly_closes, 20)
    ma_30w = calculate_simple_moving_average(weekly_closes, 30)
    ma_60w = calculate_simple_moving_average(weekly_closes, 60)
    prev_ma_30w = calculate_simple_moving_average(weekly_closes[:-1], 30)
    prev_ma_60w = calculate_simple_moving_average(weekly_closes[:-1], 60)
    weekly_cross_direction = get_weekly_cross_direction(prev_ma_30w, prev_ma_60w, ma_30w, ma_60w)
    weekly_crossed = weekly_cross_direction is not None
    weekly_bullish_crossed = weekly_cross_direction == "bullish"
    weekly_bearish_crossed = weekly_cross_direction == "bearish"
    weekly_close = weekly_closes[-1] if weekly_closes else quote.latest_price
    prev_weekly_close = weekly_closes[-2] if len(weekly_closes) >= 2 else weekly_close
    weekly_breakout_above_ma60w = has_effective_breakout_above_ma60w(weekly_bars, ma_60w, prev_ma_60w)

    triggered_labels: list[str] = []
    if ma_250 and quote.latest_price <= ma_250:
        triggered_labels.append("250日线")
    if boll_mid and quote.latest_price <= boll_mid:
        triggered_labels.append("BOLL中轨")
    if boll_lower and quote.latest_price <= boll_lower:
        triggered_labels.append("BOLL下轨")
    if boll_upper and quote.latest_price >= boll_upper:
        triggered_labels.append("BOLL上轨卖出")
    if dividend_yield >= 4.5:
        triggered_labels.append("股息率")
    if 0 < dividend_yield < SELL_DIVIDEND_THRESHOLD:
        triggered_labels.append("低股息率卖出")
    if weekly_bullish_crossed:
        triggered_labels.append("30周线上穿60周线")
    if weekly_bearish_crossed:
        triggered_labels.append("30周线下穿60周线")
    if weekly_breakout_above_ma60w:
        triggered_labels.append("有效突破60周线")

    trigger_state = "、".join(triggered_labels) if triggered_labels else "正常"
    trigger_detail = (
        f"现价 {quote.latest_price:.2f} | 涨跌 {latest_change_amount:+.2f} / {latest_change_pct:+.2f}% | 周收盘 {weekly_close:.2f} | 250日线 {ma_250:.2f} | "
        f"30周/60周 {ma_30w:.2f}/{ma_60w:.2f} | "
        f"周BOLL上/中/下轨 {boll_upper:.2f}/{boll_mid:.2f}/{boll_lower:.2f} | 股息率 {dividend_yield:.2f}% | "
        f"量化综合盈利概率 {quant_probability:.2f}%"
    )
    return SnapshotComputation(
        symbol=symbol,
        display_name=quote.name,
        latest_price=quote.latest_price,
        latest_change_amount=latest_change_amount,
        latest_change_pct=latest_change_pct,
        ma_250=ma_250,
        ma_30w=ma_30w,
        ma_60w=ma_60w,
        prev_ma_30w=prev_ma_30w,
        prev_ma_60w=prev_ma_60w,
        boll_mid=boll_mid,
        boll_lower=boll_lower,
        boll_upper=boll_upper,
        dividend_yield=dividend_yield,
        quant_probability=quant_probability,
        quant_model_breakdown=quant_model_breakdown,
        trigger_state=trigger_state,
        trigger_detail=trigger_detail,
        triggered_labels=tuple(triggered_labels),
        weekly_crossed=weekly_crossed,
        updated_at=quote.updated_at,
        weekly_bullish_crossed=weekly_bullish_crossed,
        weekly_bearish_crossed=weekly_bearish_crossed,
        weekly_close=weekly_close,
        prev_weekly_close=prev_weekly_close,
        weekly_breakout_above_ma60w=weekly_breakout_above_ma60w,
    )


class StockMonitor:
    """Coordinate time rules, market data refresh, alerts and chart payloads."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.provider = load_provider(settings.provider_name)
        try:
            self.trade_calendar = TradeCalendar(self.provider.get_trade_dates())
        except Exception as exc:
            self.trade_calendar = TradeCalendar()
            self.last_error_message = f"交易日历加载失败，已回退工作日模式：{exc}"
        self._lock = threading.Lock()
        self.last_refresh_at: datetime | None = None
        if not hasattr(self, "last_error_message"):
            self.last_error_message: str | None = None

    def run(self) -> str:
        tracked_symbols = get_monitored_stocks(self.settings.db_path)
        return f"Monitoring {len(tracked_symbols)} stock(s) with provider {self.provider.provider_name}."

    def get_market_status(self, now: datetime | None = None) -> MarketStatus:
        return get_market_status(
            refresh_interval_seconds=self.settings.refresh_interval_seconds,
            trade_calendar=self.trade_calendar,
            timezone_name=self.settings.timezone_name,
            now=now,
        )

    def _aggregate_monthly_bars(self, daily_bars: list[PriceBar]) -> list[PriceBar]:
        monthly_bars: list[PriceBar] = []
        current_bucket: list[PriceBar] = []
        current_key: tuple[int, int] | None = None

        for bar in daily_bars:
            bucket_key = (bar.traded_on.year, bar.traded_on.month)
            if current_key is None or bucket_key == current_key:
                current_bucket.append(bar)
                current_key = bucket_key
                continue
            monthly_bars.append(
                PriceBar(
                    traded_on=current_bucket[-1].traded_on,
                    open_price=current_bucket[0].open_price,
                    high_price=max(item.high_price for item in current_bucket),
                    low_price=min(item.low_price for item in current_bucket),
                    close_price=current_bucket[-1].close_price,
                )
            )
            current_bucket = [bar]
            current_key = bucket_key

        if current_bucket:
            monthly_bars.append(
                PriceBar(
                    traded_on=current_bucket[-1].traded_on,
                    open_price=current_bucket[0].open_price,
                    high_price=max(item.high_price for item in current_bucket),
                    low_price=min(item.low_price for item in current_bucket),
                    close_price=current_bucket[-1].close_price,
                )
            )
        return monthly_bars

    def _build_candlestick_payload(self, bars: list[PriceBar], limit: int, label_format: str) -> dict[str, list[float | str]]:
        visible_bars = bars[-limit:]
        return {
            "labels": [item.traded_on.strftime(label_format) for item in visible_bars],
            "open": [round(float(item.open_price), 2) for item in visible_bars],
            "high": [round(float(item.high_price), 2) for item in visible_bars],
            "low": [round(float(item.low_price), 2) for item in visible_bars],
            "close": [round(float(item.close_price), 2) for item in visible_bars],
        }

    def build_chart_payload(self, symbol: str) -> dict[str, Any]:
        daily_bars = self.provider.get_daily_bars(symbol, limit=max(self.settings.detail_chart_days * 6, 320))
        weekly_bars = self.provider.get_weekly_bars(symbol, limit=120)
        monthly_bars = self._aggregate_monthly_bars(daily_bars)
        closes = [float(item.close_price) for item in daily_bars]
        boll_limit = max(self.settings.detail_chart_days, 20)
        labels = [item.traded_on.strftime("%Y-%m-%d") for item in daily_bars[-boll_limit:]]
        close_series = closes[-boll_limit:]
        ma_250_series: list[float | None] = []
        boll_mid_series: list[float | None] = []
        boll_lower_series: list[float | None] = []
        boll_upper_series: list[float | None] = []
        for index in range(len(daily_bars)):
            ma_250_series.append(
                calculate_simple_moving_average(closes[: index + 1], 250) if index >= 249 else None
            )
            boll_mid_series.append(
                calculate_simple_moving_average(closes[: index + 1], 20) if index >= 19 else None
            )
            boll_lower_series.append(
                calculate_bollinger_lower_band(closes[: index + 1], 20) if index >= 19 else None
            )
            boll_upper_series.append(
                calculate_bollinger_upper_band(closes[: index + 1], 20) if index >= 19 else None
            )
        boll_payload = {
            "labels": labels,
            "close": [round(float(value), 2) for value in close_series],
            "ma250": ma_250_series[-boll_limit:],
            "bollMid": boll_mid_series[-boll_limit:],
            "bollLower": boll_lower_series[-boll_limit:],
            "bollUpper": boll_upper_series[-boll_limit:],
        }
        return {
            "daily_k": self._build_candlestick_payload(daily_bars, max(self.settings.detail_chart_days, 60), "%Y-%m-%d"),
            "weekly_k": self._build_candlestick_payload(weekly_bars, 52, "%Y-%m-%d"),
            "monthly_k": self._build_candlestick_payload(monthly_bars, 36, "%Y-%m"),
            "boll": boll_payload,
            "labels": boll_payload["labels"],
            "close": boll_payload["close"],
            "ma250": boll_payload["ma250"],
            "bollMid": boll_payload["bollMid"],
            "bollLower": boll_payload["bollLower"],
            "bollUpper": boll_payload["bollUpper"],
        }

    def build_snapshot(
        self,
        symbol: str,
        selected_models: list[str] | tuple[str, ...] | None = None,
        strategy_params: dict[str, float | bool] | None = None,
        live_feedback: dict[str, dict[str, float | int]] | None = None,
        force_refresh: bool = False,
    ) -> SnapshotComputation:
        """Build a snapshot for one symbol.

        盘中优先使用实时价格；如果盘后或接口短暂返回 0，
        就退回到最近一个日线收盘价，这样新增股票后在非交易时段也能立刻看到最近可用数据。
        """

        if force_refresh:
            self.provider.invalidate_symbol_cache(symbol)

        daily_bars = self.provider.get_daily_bars(symbol)
        weekly_bars = self.provider.get_weekly_bars(symbol)
        quote_error: Exception | None = None
        try:
            quote = self.provider.get_quote(symbol)
        except Exception as exc:
            quote_error = exc
            quote = self._build_quote_from_recent_bars(symbol, daily_bars, weekly_bars)

        effective_quote = quote
        if quote.latest_price <= 0 and daily_bars:
            latest_daily_bar = daily_bars[-1]
            effective_quote = Quote(
                symbol=quote.symbol,
                name=quote.name,
                latest_price=latest_daily_bar.close_price,
                updated_at=datetime.now(UTC),
            )
        if quote_error is not None:
            self.last_error_message = f"{symbol} 实时报价获取失败，已回退到最近K线价格：{quote_error}"

        dividend_yield = self.provider.get_trailing_dividend_yield(symbol, effective_quote.latest_price)
        symbol_fundamentals = self.provider.get_symbol_fundamentals(symbol)
        quant_signal = build_quant_signal(
            latest_price=effective_quote.latest_price,
            ma_250=calculate_simple_moving_average([item.close_price for item in daily_bars], 250),
            boll_mid=calculate_simple_moving_average([item.close_price for item in daily_bars], 20),
            boll_lower=calculate_bollinger_lower_band([item.close_price for item in daily_bars]),
            boll_upper=calculate_bollinger_upper_band([item.close_price for item in daily_bars]),
            ma_30w=calculate_simple_moving_average([item.close_price for item in weekly_bars], 30),
            ma_60w=calculate_simple_moving_average([item.close_price for item in weekly_bars], 60),
            dividend_yield=dividend_yield,
            daily_bars=daily_bars,
            weekly_bars=weekly_bars,
            symbol_fundamentals=symbol_fundamentals,
            live_feedback=live_feedback,
            selected_models=normalize_selected_models(selected_models),
            strategy_params=normalize_strategy_params(strategy_params),
        )
        base_snapshot = compute_snapshot_metrics(
            symbol=symbol,
            quote=effective_quote,
            daily_bars=daily_bars,
            weekly_bars=weekly_bars,
            dividend_yield=dividend_yield,
            quant_probability=quant_signal.probability,
            quant_model_breakdown=quant_signal.breakdown_json,
        )
        return replace(base_snapshot, **self._build_snapshot_context(symbol, daily_bars, base_snapshot.updated_at.date(), base_snapshot.trigger_detail))

    @staticmethod
    def _build_quote_from_recent_bars(
        symbol: str,
        daily_bars: list[PriceBar],
        weekly_bars: list[PriceBar],
    ) -> Quote:
        fallback_bar = daily_bars[-1] if daily_bars else weekly_bars[-1] if weekly_bars else None
        fallback_price = float(fallback_bar.close_price) if fallback_bar else 0.0
        return Quote(
            symbol=symbol,
            name=symbol,
            latest_price=fallback_price,
            updated_at=datetime.now(UTC),
        )

    @staticmethod
    def _estimate_position_concentration_pct(
        symbol: str,
        latest_price: float,
        position_summary: dict[str, Any] | None,
        owner_positions: dict[str, dict[str, Any]],
        total_investment_amount: float,
    ) -> float:
        if not position_summary:
            return 0.0
        quantity = int(position_summary.get("position_quantity") or 0)
        if quantity <= 0:
            return 0.0
        current_market_value = max(0.0, latest_price) * quantity
        if total_investment_amount > 0:
            return round(current_market_value / total_investment_amount * 100, 2)
        return 0.0

    def _build_snapshot_context(
        self,
        symbol: str,
        daily_bars: list[PriceBar],
        trade_date: date,
        base_trigger_detail: str,
    ) -> dict[str, Any]:
        latest_volume_ratio = calculate_volume_ratio(daily_bars)

        market_scores: list[float] = []
        for index_symbol in ("sh000001", "sz399001", "sz399006"):
            try:
                index_bars = self.provider.get_reference_index_daily_bars(index_symbol, limit=90)
            except Exception:
                index_bars = []
            if not index_bars:
                continue
            _, score = classify_trend_environment(index_bars)
            market_scores.append(score)
        market_bias_score = round(sum(market_scores) / len(market_scores), 2) if market_scores else 0.0
        if market_bias_score >= 18.0:
            market_environment = "偏强"
        elif market_bias_score <= -18.0:
            market_environment = "偏弱"
        else:
            market_environment = "中性"

        try:
            profile = self.provider.get_symbol_profile(symbol) or {}
        except Exception:
            profile = {}
        industry_name = str(profile.get("industry_name") or "")
        try:
            industry_bars = self.provider.get_industry_daily_bars(industry_name, limit=90) if industry_name else []
        except Exception:
            industry_bars = []
        industry_environment, industry_bias_score = classify_trend_environment(industry_bars)
        earnings_phase, earnings_days_to_window = infer_earnings_phase(trade_date)
        trigger_detail = (
            f"{base_trigger_detail} | 量能比 {latest_volume_ratio:.2f} | 大盘 {market_environment}({market_bias_score:.0f})"
            f" | 行业 {industry_name or '-'} {industry_environment}({industry_bias_score:.0f}) | 财报节奏 {earnings_phase}"
        )
        return {
            "latest_volume_ratio": latest_volume_ratio,
            "market_environment": market_environment,
            "market_bias_score": market_bias_score,
            "industry_name": industry_name,
            "industry_environment": industry_environment,
            "industry_bias_score": industry_bias_score,
            "earnings_phase": earnings_phase,
            "earnings_days_to_window": earnings_days_to_window,
            "trigger_detail": trigger_detail,
        }

    def refresh_symbol_snapshot(
        self,
        owner_username: str,
        symbol: str,
        force_refresh: bool = False,
    ) -> SnapshotComputation:
        quant_config = self._resolve_owner_quant_config(owner_username)
        try:
            snapshot = self.build_snapshot(
                symbol,
                selected_models=quant_config["selected_models"],
                strategy_params=quant_config["strategy_params"],
                live_feedback=quant_config["paper_trade_feedback"],
                force_refresh=force_refresh,
            )
        except Exception as exc:
            fallback_row = get_snapshot(self.settings.db_path, owner_username, symbol)
            if fallback_row is None:
                raise
            self.last_error_message = f"{owner_username}/{symbol} 刷新失败，已保留最近快照：{exc}"
            self.last_refresh_at = datetime.now(UTC)
            return self._snapshot_from_row(fallback_row)
        upsert_snapshot(
            db_path=self.settings.db_path,
            owner_username=owner_username,
            symbol=snapshot.symbol,
            display_name=snapshot.display_name,
            latest_price=snapshot.latest_price,
            latest_change_amount=snapshot.latest_change_amount,
            latest_change_pct=snapshot.latest_change_pct,
            ma_250=snapshot.ma_250,
            ma_30w=snapshot.ma_30w,
            ma_60w=snapshot.ma_60w,
            boll_mid=snapshot.boll_mid,
            boll_lower=snapshot.boll_lower,
            boll_upper=snapshot.boll_upper,
            dividend_yield=snapshot.dividend_yield,
            quant_probability=snapshot.quant_probability,
            quant_model_breakdown=snapshot.quant_model_breakdown,
            trigger_state=snapshot.trigger_state,
            trigger_detail=snapshot.trigger_detail,
            latest_volume_ratio=snapshot.latest_volume_ratio,
            market_environment=snapshot.market_environment,
            market_bias_score=snapshot.market_bias_score,
            industry_name=snapshot.industry_name,
            industry_environment=snapshot.industry_environment,
            industry_bias_score=snapshot.industry_bias_score,
            earnings_phase=snapshot.earnings_phase,
            earnings_days_to_window=snapshot.earnings_days_to_window,
            updated_at=snapshot.updated_at.isoformat(),
        )
        self._sync_model_paper_trades(owner_username, snapshot)
        self.last_refresh_at = datetime.now(UTC)
        return snapshot

    @staticmethod
    def _snapshot_from_row(row: Any) -> SnapshotComputation:
        updated_at_raw = row["updated_at"] if row["updated_at"] else datetime.now(UTC).isoformat()
        trigger_state = str(row["trigger_state"] or "正常")
        triggered_labels = tuple(item for item in trigger_state.split("、") if item and item != "正常")
        return SnapshotComputation(
            symbol=str(row["symbol"]),
            display_name=str(row["display_name"] or row["symbol"]),
            latest_price=float(row["latest_price"] or 0.0),
            latest_change_amount=float(row["latest_change_amount"] or 0.0),
            latest_change_pct=float(row["latest_change_pct"] or 0.0),
            ma_250=float(row["ma_250"] or 0.0),
            ma_30w=float(row["ma_30w"] or 0.0),
            ma_60w=float(row["ma_60w"] or 0.0),
            prev_ma_30w=float(row["ma_30w"] or 0.0),
            prev_ma_60w=float(row["ma_60w"] or 0.0),
            boll_mid=float(row["boll_mid"] or 0.0),
            boll_lower=float(row["boll_lower"] or 0.0),
            boll_upper=float(row["boll_upper"] or 0.0),
            dividend_yield=float(row["dividend_yield"] or 0.0),
            quant_probability=float(row["quant_probability"] or 0.0),
            quant_model_breakdown=str(row["quant_model_breakdown"] or "[]"),
            trigger_state=trigger_state,
            trigger_detail=str(row["trigger_detail"] or ""),
            triggered_labels=triggered_labels,
            weekly_crossed=any(item in {"30周/60周均线", "30周线上穿60周线", "30周线下穿60周线"} for item in triggered_labels),
            updated_at=datetime.fromisoformat(updated_at_raw),
            latest_volume_ratio=float(row["latest_volume_ratio"] or 1.0),
            market_environment=str(row["market_environment"] or "中性"),
            market_bias_score=float(row["market_bias_score"] or 0.0),
            industry_name=str(row["industry_name"] or ""),
            industry_environment=str(row["industry_environment"] or "中性"),
            industry_bias_score=float(row["industry_bias_score"] or 0.0),
            earnings_phase=str(row["earnings_phase"] or "常规窗口"),
            earnings_days_to_window=int(row["earnings_days_to_window"] or 999),
            weekly_bullish_crossed="30周线上穿60周线" in triggered_labels,
            weekly_bearish_crossed="30周线下穿60周线" in triggered_labels,
            weekly_breakout_above_ma60w="有效突破60周线" in triggered_labels,
        )

    def run_cycle(self, now: datetime | None = None) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            status = self.get_market_status(now)
            trade_date = status.current_time.date()
            if status.is_pre_open_window:
                self._prepare_dividend_alerts(trade_date)
            if status.is_market_open:
                self._deliver_pending_alerts(trade_date)
                self._refresh_open_session(trade_date)
            if status.is_post_close:
                self._send_post_close_holding_reviews(trade_date)
                if self.trade_calendar.is_last_trading_day_of_week(trade_date):
                    self._prepare_weekly_cross_alerts(trade_date)
        except Exception as exc:  # pragma: no cover
            self.last_error_message = str(exc)
        finally:
            self._lock.release()

    def _refresh_open_session(self, trade_date: date) -> None:
        snapshot_cache: dict[tuple[str, tuple[str, ...], str], SnapshotComputation] = {}
        owner_position_cache: dict[str, dict[str, dict[str, Any]]] = {}
        owner_portfolio_base_cache: dict[str, float] = {}
        owner_email_cache: dict[str, Any] = {}
        owner_quant_cache: dict[str, dict[str, Any]] = {}
        trade_marker = trade_date.isoformat()
        for stock in get_monitored_stocks(self.settings.db_path):
            owner_username = stock["owner_username"]
            symbol = stock["symbol"]
            try:
                email_settings = owner_email_cache.setdefault(
                    owner_username,
                    get_email_settings(self.settings.db_path, owner_username),
                )
                quant_settings = owner_quant_cache.setdefault(
                    owner_username,
                    self._resolve_owner_quant_config(owner_username),
                )
                owner_positions = owner_position_cache.setdefault(
                    owner_username,
                    self._resolve_owner_position_map(owner_username),
                )
                owner_total_investment = owner_portfolio_base_cache.setdefault(
                    owner_username,
                    float(get_portfolio_settings(self.settings.db_path, owner_username)["total_investment_amount"] or 0.0),
                )
                selected_models = quant_settings["selected_models"]
                strategy_params = quant_settings["strategy_params"]
                cache_key = (
                    symbol,
                    tuple(selected_models),
                    json.dumps(strategy_params, ensure_ascii=False, sort_keys=True),
                    json.dumps(quant_settings["paper_trade_feedback"], ensure_ascii=False, sort_keys=True),
                )
                snapshot = snapshot_cache.get(cache_key)
                if snapshot is None:
                    snapshot = self.build_snapshot(
                        symbol,
                        selected_models=selected_models,
                        strategy_params=strategy_params,
                        live_feedback=quant_settings["paper_trade_feedback"],
                    )
                    snapshot_cache[cache_key] = snapshot
                upsert_snapshot(
                    db_path=self.settings.db_path,
                    owner_username=owner_username,
                    symbol=snapshot.symbol,
                    display_name=snapshot.display_name,
                    latest_price=snapshot.latest_price,
                    latest_change_amount=snapshot.latest_change_amount,
                    latest_change_pct=snapshot.latest_change_pct,
                    ma_250=snapshot.ma_250,
                    ma_30w=snapshot.ma_30w,
                    ma_60w=snapshot.ma_60w,
                    boll_mid=snapshot.boll_mid,
                    boll_lower=snapshot.boll_lower,
                    boll_upper=snapshot.boll_upper,
                    dividend_yield=snapshot.dividend_yield,
                    quant_probability=snapshot.quant_probability,
                    quant_model_breakdown=snapshot.quant_model_breakdown,
                    trigger_state=snapshot.trigger_state,
                    trigger_detail=snapshot.trigger_detail,
                    latest_volume_ratio=snapshot.latest_volume_ratio,
                    market_environment=snapshot.market_environment,
                    market_bias_score=snapshot.market_bias_score,
                    industry_name=snapshot.industry_name,
                    industry_environment=snapshot.industry_environment,
                    industry_bias_score=snapshot.industry_bias_score,
                    earnings_phase=snapshot.earnings_phase,
                    earnings_days_to_window=snapshot.earnings_days_to_window,
                    updated_at=snapshot.updated_at.isoformat(),
                )
                position_summary = owner_positions.get(symbol)
                position_concentration_pct = self._estimate_position_concentration_pct(
                    symbol,
                    snapshot.latest_price,
                    position_summary,
                    owner_positions,
                    owner_total_investment,
                )
                snapshot = replace(snapshot, position_concentration_pct=position_concentration_pct)
                stock_advice = build_stock_comprehensive_advice(snapshot.__dict__)
                stock_advice["position_concentration_pct"] = position_concentration_pct
                self._sync_model_paper_trades(owner_username, snapshot)
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="250日线",
                    trade_marker=trade_marker,
                    condition_met=bool(snapshot.ma_250 and snapshot.latest_price <= snapshot.ma_250 and self._should_emit_buy_alert(snapshot, "250日线", stock_advice)),
                    detail=snapshot.trigger_detail,
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "ma_250": snapshot.ma_250,
                        "latest_price": snapshot.latest_price,
                    },
                    email_settings=email_settings,
                    required_hits=1,
                )
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="BOLL中轨",
                    trade_marker=trade_marker,
                    condition_met=bool(snapshot.boll_mid and snapshot.latest_price <= snapshot.boll_mid and self._should_emit_buy_alert(snapshot, "BOLL中轨", stock_advice)),
                    detail=f"{snapshot.trigger_detail} | 周线BOLL：盘中现价回落至周BOLL中轨下方。",
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "boll_mid": snapshot.boll_mid,
                        "latest_price": snapshot.latest_price,
                    },
                    email_settings=email_settings,
                    required_hits=1,
                )
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="BOLL下轨",
                    trade_marker=trade_marker,
                    condition_met=bool(snapshot.boll_lower and snapshot.latest_price <= snapshot.boll_lower and self._should_emit_buy_alert(snapshot, "BOLL下轨", stock_advice)),
                    detail=f"{snapshot.trigger_detail} | 周线BOLL：盘中现价触及周BOLL下轨。",
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "boll_lower": snapshot.boll_lower,
                        "latest_price": snapshot.latest_price,
                    },
                    email_settings=email_settings,
                    required_hits=1,
                )
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="BOLL上轨卖出",
                    trade_marker=trade_marker,
                    condition_met=bool(snapshot.boll_upper and snapshot.latest_price >= snapshot.boll_upper and self._should_emit_sell_alert(position_summary, snapshot, "BOLL上轨卖出", stock_advice)),
                    detail=f"{snapshot.trigger_detail} | 周线BOLL：盘中价格突破周BOLL上轨，可结合仓位分批止盈。",
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "boll_upper": snapshot.boll_upper,
                        "latest_price": snapshot.latest_price,
                    },
                    email_settings=email_settings,
                    required_hits=2,
                )
                threshold = float(quant_settings["probability_threshold"])
                if bool(quant_settings["enabled"]):
                    self._process_intraday_signal(
                        owner_username=owner_username,
                        symbol=symbol,
                        display_name=snapshot.display_name,
                        trigger_type="量化盈利概率",
                        trade_marker=trade_marker,
                        condition_met=bool(snapshot.quant_probability >= threshold and self._should_emit_buy_alert(snapshot, "量化盈利概率", stock_advice)),
                        detail=f"{snapshot.trigger_detail} | 量化阈值 {threshold:.2f}%",
                        current_price=snapshot.latest_price,
                        indicator_values={
                            "quant_probability": snapshot.quant_probability,
                            "probability_threshold": threshold,
                            "selected_models": ", ".join(selected_models),
                        },
                        email_settings=email_settings,
                        required_hits=1,
                    )
                    quant_sell_signal = build_quant_sell_signal(snapshot)
                    self._process_intraday_signal(
                        owner_username=owner_username,
                        symbol=symbol,
                        display_name=snapshot.display_name,
                        trigger_type="量化走弱卖出",
                        trade_marker=trade_marker,
                        condition_met=bool(quant_sell_signal.should_alert and self._should_emit_sell_alert(position_summary, snapshot, "量化走弱卖出", stock_advice)),
                        detail=f"{snapshot.trigger_detail} | 卖出参考：{'；'.join(quant_sell_signal.reasons)}",
                        current_price=snapshot.latest_price,
                        indicator_values={
                            "quant_probability": snapshot.quant_probability,
                            "sell_probability_threshold": QUANT_SELL_PROBABILITY_THRESHOLD,
                            "confirmation_count": quant_sell_signal.confirmation_count,
                            "reasons": "；".join(quant_sell_signal.reasons),
                        },
                        email_settings=email_settings,
                        required_hits=1,
                    )
                self.last_refresh_at = datetime.now(UTC)
                self.last_error_message = None
            except Exception as exc:
                self.last_error_message = f"{owner_username}/{symbol} 刷新失败：{exc}"

    def _resolve_owner_position_map(self, owner_username: str) -> dict[str, dict[str, Any]]:
        position_map: dict[str, dict[str, Any]] = {}
        symbols = {str(row["symbol"]) for row in list_trade_records(self.settings.db_path, owner_username, None)}
        for symbol in symbols:
            trades = list_trade_records_for_symbol(self.settings.db_path, owner_username, symbol)
            if not trades:
                continue
            position_summary = build_position_summary(trades)
            if int(position_summary.get("position_quantity") or 0) > 0:
                position_map[symbol] = position_summary
        return position_map

    @staticmethod
    def _buy_expected_upside_pct(current_price: float, stock_advice: dict[str, Any]) -> float:
        suggested_reduce_price = float(stock_advice.get("suggested_reduce_price") or 0.0)
        if current_price <= 0 or suggested_reduce_price <= current_price:
            return 0.0
        return (suggested_reduce_price - current_price) / current_price * 100

    @staticmethod
    def _should_emit_buy_alert(
        snapshot: SnapshotComputation,
        trigger_type: str,
        stock_advice: dict[str, Any],
    ) -> bool:
        if trigger_type not in BUY_TRIGGER_TYPES:
            return False
        if stock_advice.get("action") != "偏买入":
            return False

        triggered_labels = set(snapshot.triggered_labels)
        if trigger_type != "量化盈利概率" and trigger_type not in triggered_labels:
            return False

        buy_level = int(stock_advice.get("buy_recommendation_level") or 0)
        sell_level = int(stock_advice.get("sell_recommendation_level") or 0)
        quant_probability = float(snapshot.quant_probability or 0.0)
        sell_signals = set(stock_advice.get("sell_signals") or ())
        support_triggers = triggered_labels & PRICE_SUPPORT_TRIGGER_TYPES
        has_dividend_support = "股息率" in triggered_labels
        has_weekly_support = bool(triggered_labels & WEEKLY_SUPPORT_TRIGGER_TYPES)
        top_model_dispersion = float(stock_advice.get("top_model_dispersion") or 0.0)
        expected_upside_pct = StockMonitor._buy_expected_upside_pct(snapshot.latest_price, stock_advice)
        market_environment = str(snapshot.market_environment or "中性")
        industry_environment = str(snapshot.industry_environment or "中性")
        latest_volume_ratio = float(snapshot.latest_volume_ratio or 1.0)
        earnings_phase = str(snapshot.earnings_phase or "常规窗口")
        position_concentration_pct = float(stock_advice.get("position_concentration_pct") or snapshot.position_concentration_pct or 0.0)
        dcf_gap_pct_raw = stock_advice.get("dcf_valuation_gap_pct")
        try:
            dcf_gap_pct = float(dcf_gap_pct_raw) if dcf_gap_pct_raw is not None else None
        except (TypeError, ValueError):
            dcf_gap_pct = None

        if buy_level < BUY_ALERT_MIN_LEVEL:
            return False
        if buy_level - sell_level < BUY_ALERT_MIN_LEVEL_GAP:
            return False
        if quant_probability < BUY_ALERT_MIN_QUANT_PROBABILITY:
            return False
        if sell_signals:
            return False
        if top_model_dispersion > BUY_ALERT_MAX_MODEL_DISPERSION:
            return False
        if expected_upside_pct < BUY_ALERT_MIN_EXPECTED_UPSIDE_PCT:
            return False
        if dcf_gap_pct is not None and dcf_gap_pct < BUY_ALERT_MIN_DCF_GAP_PCT:
            return False
        if latest_volume_ratio < 0.85:
            return False
        if market_environment == "偏弱" and quant_probability < BUY_ALERT_DIVIDEND_CONFIRMATION_QUANT:
            return False
        if market_environment == "偏弱" and industry_environment == "偏弱":
            return False
        if industry_environment == "偏弱" and quant_probability < BUY_ALERT_STRONG_QUANT_PROBABILITY:
            return False
        if earnings_phase != "常规窗口" and quant_probability < BUY_ALERT_STRONG_QUANT_PROBABILITY:
            return False
        if position_concentration_pct >= 30 and quant_probability < 88:
            return False
        if position_concentration_pct >= 45:
            return False

        if trigger_type == "量化盈利概率":
            return bool(len(support_triggers) >= 2 or has_dividend_support or has_weekly_support)
        if trigger_type in PRICE_SUPPORT_TRIGGER_TYPES:
            return (
                len(support_triggers) >= 2
                or has_dividend_support
                or has_weekly_support
                or quant_probability >= BUY_ALERT_STRONG_QUANT_PROBABILITY
            )
        if trigger_type == "股息率":
            return bool(
                support_triggers
                or has_weekly_support
                or quant_probability >= BUY_ALERT_DIVIDEND_CONFIRMATION_QUANT
            )
        if trigger_type in WEEKLY_SUPPORT_TRIGGER_TYPES:
            return bool(
                support_triggers
                or has_dividend_support
                or quant_probability >= BUY_ALERT_WEEKLY_CONFIRMATION_QUANT
            )
        return False

    @staticmethod
    def _should_emit_sell_alert(
        position_summary: dict[str, Any] | None,
        snapshot: SnapshotComputation,
        trigger_type: str,
        stock_advice: dict[str, Any],
    ) -> bool:
        if trigger_type not in SELL_TRIGGER_TYPES:
            return False
        if not position_summary or int(position_summary.get("position_quantity") or 0) <= 0:
            return False

        action = str(stock_advice.get("action") or "观望")
        buy_level = int(stock_advice.get("buy_recommendation_level") or 0)
        sell_level = int(stock_advice.get("sell_recommendation_level") or 0)
        sell_signals = set(stock_advice.get("sell_signals") or ())
        quant_probability = float(snapshot.quant_probability or 0.0)
        position_concentration_pct = float(stock_advice.get("position_concentration_pct") or snapshot.position_concentration_pct or 0.0)

        if trigger_type == "BOLL上轨卖出":
            extension_ratio = snapshot.latest_price / snapshot.boll_upper if snapshot.boll_upper > 0 else 1.0
            required_extension = SELL_ALERT_BOLL_EXTENSION_RATIO - (0.01 if position_concentration_pct >= 30 else 0.0)
            return (
                trigger_type in snapshot.triggered_labels
                and (
                    extension_ratio >= required_extension
                    or action == "偏卖出"
                    or sell_level >= SELL_ALERT_STRONG_LEVEL
                    or "量化走弱卖出" in sell_signals
                    or position_concentration_pct >= 35
                )
            )
        if trigger_type == "低股息率卖出":
            return (
                trigger_type in snapshot.triggered_labels
                and (
                    action == "偏卖出"
                    or sell_level >= SELL_ALERT_STRONG_LEVEL
                    or quant_probability <= SELL_ALERT_LOW_DIVIDEND_MAX_QUANT
                    or position_concentration_pct >= 35
                )
            )
        if trigger_type == "量化走弱卖出":
            return (
                sell_level >= SELL_ALERT_MIN_LEVEL
                and (
                    action == "偏卖出"
                    or sell_level >= max(SELL_ALERT_STRONG_LEVEL - 1, buy_level)
                    or len(sell_signals) >= 2
                    or position_concentration_pct >= 30
                )
            )
        return False

    def _send_post_close_holding_reviews(self, trade_date: date) -> None:
        marker = trade_date.isoformat()
        owners = sorted({row["owner_username"] for row in get_monitored_stocks(self.settings.db_path)})
        for owner_username in owners:
            job_state = get_job_state(self.settings.db_path, owner_username, "post_close_holding_review")
            if job_state and job_state["last_run_marker"] == marker:
                continue
            position_map = self._resolve_owner_position_map(owner_username)
            if not position_map:
                set_job_state(self.settings.db_path, owner_username, "post_close_holding_review", marker)
                continue
            email_settings = get_email_settings(self.settings.db_path, owner_username)
            snapshots: list[dict[str, Any]] = []
            for symbol in sorted(position_map):
                try:
                    snapshot = self.refresh_symbol_snapshot(owner_username, symbol)
                except Exception as exc:
                    fallback_row = get_snapshot(self.settings.db_path, owner_username, symbol)
                    if fallback_row is None:
                        self.last_error_message = f"{owner_username}/{symbol} 收盘复盘快照刷新失败：{exc}"
                        continue
                    snapshot = self._snapshot_from_row(fallback_row)
                    self.last_error_message = f"{owner_username}/{symbol} 收盘复盘沿用最近快照：{exc}"
                snapshots.append(self._snapshot_payload(snapshot))
            if not snapshots:
                set_job_state(self.settings.db_path, owner_username, "post_close_holding_review", marker)
                continue
            portfolio_settings = get_portfolio_settings(self.settings.db_path, owner_username)
            portfolio_profile = build_portfolio_profile(
                list_trade_records(self.settings.db_path, owner_username, None),
                snapshots,
                float(portfolio_settings["total_investment_amount"] or 0.0),
            )
            model_learning = self._build_model_learning_summary(owner_username, snapshots)
            email_result = send_message(
                email_settings,
                subject=f"[收盘持仓复盘] {owner_username} {marker}",
                body=build_portfolio_review_email_body(
                    {
                        "owner_username": owner_username,
                        "trade_date": marker,
                        "portfolio_profile": portfolio_profile,
                        "model_learning": model_learning,
                    }
                ),
                html_body=build_portfolio_review_email_html_body(
                    {
                        "owner_username": owner_username,
                        "trade_date": marker,
                        "portfolio_profile": portfolio_profile,
                        "model_learning": model_learning,
                    }
                ),
            )
            if not email_result.success:
                self.last_error_message = f"{owner_username} 收盘持仓复盘邮件发送失败：{email_result.error}"
            set_job_state(self.settings.db_path, owner_username, "post_close_holding_review", marker)

    @staticmethod
    def _snapshot_payload(snapshot: SnapshotComputation) -> dict[str, Any]:
        return {
            "symbol": snapshot.symbol,
            "display_name": snapshot.display_name,
            "latest_price": snapshot.latest_price,
            "latest_change_amount": snapshot.latest_change_amount,
            "latest_change_pct": snapshot.latest_change_pct,
            "ma_250": snapshot.ma_250,
            "ma_30w": snapshot.ma_30w,
            "ma_60w": snapshot.ma_60w,
            "boll_mid": snapshot.boll_mid,
            "boll_lower": snapshot.boll_lower,
            "boll_upper": snapshot.boll_upper,
            "dividend_yield": snapshot.dividend_yield,
            "quant_probability": snapshot.quant_probability,
            "quant_model_breakdown": snapshot.quant_model_breakdown,
            "trigger_state": snapshot.trigger_state,
            "trigger_detail": snapshot.trigger_detail,
            "latest_volume_ratio": snapshot.latest_volume_ratio,
            "market_environment": snapshot.market_environment,
            "market_bias_score": snapshot.market_bias_score,
            "industry_name": snapshot.industry_name,
            "industry_environment": snapshot.industry_environment,
            "industry_bias_score": snapshot.industry_bias_score,
            "earnings_phase": snapshot.earnings_phase,
            "earnings_days_to_window": snapshot.earnings_days_to_window,
            "updated_at": snapshot.updated_at.isoformat(),
        }

    def _resolve_owner_quant_config(self, owner_username: str) -> dict[str, Any]:
        """Load one user's quant configuration and normalize it for downstream use."""

        settings = get_quant_settings(self.settings.db_path, owner_username)
        return {
            "enabled": bool(settings["enabled"]),
            "probability_threshold": float(settings["probability_threshold"]),
            "selected_models": tuple(json.loads(settings["selected_models"] or "[]")) or DEFAULT_QUANT_MODELS,
            "strategy_params": normalize_strategy_params(json.loads(settings["strategy_params"] or "{}")),
            "paper_trade_feedback": self._resolve_owner_model_feedback(owner_username),
        }

    def _resolve_owner_model_feedback(self, owner_username: str, lookback_days: int = 180) -> dict[str, dict[str, float | int]]:
        closed_rows = list_model_paper_trades(
            self.settings.db_path,
            owner_username,
            status="closed",
            date_from=(datetime.now(UTC).date() - timedelta(days=lookback_days)).isoformat(),
        )
        grouped: dict[str, list[float]] = {}
        drawdowns: dict[str, list[float]] = {}
        for row in closed_rows:
            key = str(row["model_key"])
            realized_return = float(row["realized_return_pct"] or 0.0)
            grouped.setdefault(key, []).append(realized_return)
            drawdowns.setdefault(key, []).append(float(row["max_drawdown_pct"] or 0.0))
        feedback: dict[str, dict[str, float | int]] = {}
        for key, returns in grouped.items():
            sample_size = len(returns)
            if sample_size <= 0:
                continue
            feedback[key] = {
                "sample_size": sample_size,
                "hit_rate": round(sum(1 for value in returns if value > 0) / sample_size * 100, 2),
                "avg_return_pct": round(sum(returns) / sample_size, 2),
                "avg_drawdown_pct": round(sum(drawdowns.get(key, [0.0])) / sample_size, 2),
            }
        return feedback

    @staticmethod
    def _feedback_group_weight(feedback: dict[str, float | int] | None) -> float:
        if not feedback:
            return 1.0
        sample_size = int(feedback.get("sample_size") or 0)
        if sample_size < 3:
            return 1.0
        hit_rate = float(feedback.get("hit_rate") or 0.0)
        avg_return_pct = float(feedback.get("avg_return_pct") or 0.0)
        confidence = min(1.0, sample_size / 12.0)
        raw_weight = 1.0 + ((hit_rate / 100.0) - 0.5) * 0.45 + (avg_return_pct / 100.0) * 1.3
        return round(max(0.85, min(1.35, 1.0 + (raw_weight - 1.0) * confidence)), 3)

    @staticmethod
    def _describe_learning_status(sample_size: int, hit_rate: float, avg_return_pct: float) -> str:
        if sample_size < 3:
            return "样本积累中"
        if hit_rate >= 60.0 and avg_return_pct > 0:
            return "学习有效"
        if hit_rate >= 50.0 or avg_return_pct >= 0:
            return "温和有效"
        return "待继续纠偏"

    @staticmethod
    def _describe_impact_degree(contribution_pct: float, calibration_weight: float, sample_size: int) -> str:
        if sample_size < 3 and contribution_pct < 52.0:
            return "观察中"
        if contribution_pct >= 55.0 or calibration_weight >= 1.12:
            return "高"
        if contribution_pct >= 45.0 or calibration_weight >= 1.0:
            return "中"
        return "低"

    def _build_model_learning_summary(
        self,
        owner_username: str,
        snapshots: list[dict[str, Any]],
        lookback_days: int = 180,
    ) -> dict[str, Any]:
        date_from = (datetime.now(UTC).date() - timedelta(days=lookback_days)).isoformat()
        rows = list_model_paper_trades(
            self.settings.db_path,
            owner_username,
            date_from=date_from,
        )
        summary_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = dict(row)
            key = str(payload.get("model_key") or "")
            if not key:
                continue
            summary = summary_map.setdefault(
                key,
                {
                    "model_key": key,
                    "model_label": str(payload.get("model_label") or key),
                    "model_scope": str(payload.get("model_scope") or "model"),
                    "closed_count": 0,
                    "open_count": 0,
                    "win_count": 0,
                    "return_sum": 0.0,
                    "max_drawdown_pct": 0.0,
                },
            )
            status = str(payload.get("status") or "")
            summary["max_drawdown_pct"] = max(
                float(summary["max_drawdown_pct"]),
                float(payload.get("max_drawdown_pct") or 0.0),
            )
            if status == "closed":
                realized_return_pct = float(payload.get("realized_return_pct") or 0.0)
                summary["closed_count"] = int(summary["closed_count"]) + 1
                summary["return_sum"] = float(summary["return_sum"]) + realized_return_pct
                if realized_return_pct > 0:
                    summary["win_count"] = int(summary["win_count"]) + 1
            elif status == "open":
                summary["open_count"] = int(summary["open_count"]) + 1

        current_group_scores: dict[str, list[float]] = {"professional": [], "adaptive": []}
        for snapshot in snapshots:
            try:
                breakdown = json.loads(str(snapshot.get("quant_model_breakdown") or "[]"))
            except json.JSONDecodeError:
                breakdown = []
            professional_scores = [
                float(item.get("score") or 0.0)
                for item in breakdown
                if isinstance(item, dict) and str(item.get("key") or "") in PROFESSIONAL_MODEL_KEYS
            ]
            adaptive_scores = [
                float(item.get("score") or 0.0)
                for item in breakdown
                if isinstance(item, dict) and str(item.get("key") or "") not in PROFESSIONAL_MODEL_KEYS
            ]
            if professional_scores:
                current_group_scores["professional"].append(sum(professional_scores) / len(professional_scores))
            if adaptive_scores:
                current_group_scores["adaptive"].append(sum(adaptive_scores) / len(adaptive_scores))

        groups: list[dict[str, Any]] = []
        influence_scores: dict[str, float] = {}
        for key, label in (("professional", "专业组"), ("adaptive", "自适应组")):
            item = summary_map.get(
                key,
                {
                    "model_label": label,
                    "model_scope": "group",
                    "closed_count": 0,
                    "open_count": 0,
                    "win_count": 0,
                    "return_sum": 0.0,
                    "max_drawdown_pct": 0.0,
                },
            )
            sample_size = int(item.get("closed_count") or 0)
            hit_rate = round(int(item.get("win_count") or 0) / max(sample_size, 1) * 100, 2) if sample_size else 0.0
            avg_return_pct = round(float(item.get("return_sum") or 0.0) / max(sample_size, 1), 2) if sample_size else 0.0
            current_score = round(
                sum(current_group_scores[key]) / len(current_group_scores[key]),
                2,
            ) if current_group_scores[key] else 0.0
            calibration_weight = self._feedback_group_weight(
                {
                    "sample_size": sample_size,
                    "hit_rate": hit_rate,
                    "avg_return_pct": avg_return_pct,
                }
            )
            influence_scores[key] = current_score * calibration_weight if current_score > 0 else 0.0
            groups.append(
                {
                    "key": key,
                    "label": label,
                    "sample_size": sample_size,
                    "open_count": int(item.get("open_count") or 0),
                    "hit_rate": hit_rate,
                    "avg_return_pct": avg_return_pct,
                    "max_drawdown_pct": round(float(item.get("max_drawdown_pct") or 0.0), 2),
                    "current_score": current_score,
                    "calibration_weight": calibration_weight,
                    "learning_status": self._describe_learning_status(sample_size, hit_rate, avg_return_pct),
                }
            )

        total_influence = sum(influence_scores.values())
        for item in groups:
            contribution_pct = round(
                influence_scores.get(str(item["key"]), 0.0) / total_influence * 100,
                2,
            ) if total_influence > 0 else 0.0
            item["contribution_pct"] = contribution_pct
            item["impact_degree"] = self._describe_impact_degree(
                contribution_pct,
                float(item["calibration_weight"]),
                int(item["sample_size"]),
            )
            item["impact_summary"] = (
                f"当前组内平均分 {float(item['current_score']):.2f}，"
                f"对今日最终量化结论贡献约 {contribution_pct:.2f}% ，"
                f"纸面交易调权系数 {float(item['calibration_weight']):.2f}。"
            )

        model_rows: list[dict[str, Any]] = []
        for item in summary_map.values():
            if str(item.get("model_scope") or "") != "model":
                continue
            sample_size = int(item.get("closed_count") or 0)
            if sample_size <= 0:
                continue
            hit_rate = round(int(item.get("win_count") or 0) / sample_size * 100, 2)
            avg_return_pct = round(float(item.get("return_sum") or 0.0) / sample_size, 2)
            model_rows.append(
                {
                    "key": str(item.get("model_key") or ""),
                    "label": str(item.get("model_label") or item.get("model_key") or ""),
                    "sample_size": sample_size,
                    "open_count": int(item.get("open_count") or 0),
                    "hit_rate": hit_rate,
                    "avg_return_pct": avg_return_pct,
                    "max_drawdown_pct": round(float(item.get("max_drawdown_pct") or 0.0), 2),
                    "learning_status": self._describe_learning_status(sample_size, hit_rate, avg_return_pct),
                }
            )
        model_rows.sort(
            key=lambda item: (
                -float(item["avg_return_pct"]),
                -float(item["hit_rate"]),
                -int(item["sample_size"]),
                str(item["label"]),
            )
        )

        overview_lines: list[str] = []
        if any(int(item["sample_size"]) > 0 for item in groups):
            leader = max(
                groups,
                key=lambda item: (
                    float(item["avg_return_pct"]),
                    float(item["hit_rate"]),
                    int(item["sample_size"]),
                ),
            )
            overview_lines.append(
                f"近 {lookback_days} 天纸面交易里，{leader['label']}暂时学习领先，当前状态为{leader['learning_status']}。"
            )
        if total_influence > 0:
            dominant = max(groups, key=lambda item: float(item.get("contribution_pct") or 0.0))
            overview_lines.append(
                f"今日持仓决策里，{dominant['label']}当前影响更大，贡献约 {float(dominant['contribution_pct']):.2f}% 。"
            )
        if not overview_lines:
            overview_lines.append("两套模型的纸面交易样本还在积累中，当前先以实时信号强弱为主。")

        return {
            "lookback_days": lookback_days,
            "groups": groups,
            "top_models": model_rows[:3],
            "overview_lines": overview_lines,
        }

    @staticmethod
    def _load_quant_breakdown(snapshot: SnapshotComputation) -> list[dict[str, Any]]:
        try:
            payload = json.loads(snapshot.quant_model_breakdown or "[]")
        except json.JSONDecodeError:
            return []
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _build_paper_trade_candidates(snapshot: SnapshotComputation) -> list[dict[str, Any]]:
        breakdown = StockMonitor._load_quant_breakdown(snapshot)
        professional_scores = [
            float(item.get("score") or 0.0)
            for item in breakdown
            if str(item.get("key") or "") in {"msci_momentum", "quality_stability"}
        ]
        adaptive_scores = [
            float(item.get("score") or 0.0)
            for item in breakdown
            if str(item.get("key") or "") not in {"msci_momentum", "quality_stability"}
        ]
        candidates: list[dict[str, Any]] = []
        if professional_scores:
            candidates.append(
                {
                    "model_scope": "group",
                    "model_key": "professional",
                    "model_label": "专业组",
                    "score": round(sum(professional_scores) / len(professional_scores), 2),
                    "open_threshold": PAPER_TRADE_OPEN_THRESHOLD_GROUP,
                    "close_threshold": PAPER_TRADE_CLOSE_THRESHOLD_GROUP,
                }
            )
        if adaptive_scores:
            candidates.append(
                {
                    "model_scope": "group",
                    "model_key": "adaptive",
                    "model_label": "自适应组",
                    "score": round(sum(adaptive_scores) / len(adaptive_scores), 2),
                    "open_threshold": PAPER_TRADE_OPEN_THRESHOLD_GROUP,
                    "close_threshold": PAPER_TRADE_CLOSE_THRESHOLD_GROUP,
                }
            )
        for item in breakdown:
            candidates.append(
                {
                    "model_scope": "model",
                    "model_key": str(item.get("key") or ""),
                    "model_label": str(item.get("label") or "模型"),
                    "score": float(item.get("score") or 0.0),
                    "open_threshold": PAPER_TRADE_OPEN_THRESHOLD_MODEL,
                    "close_threshold": PAPER_TRADE_CLOSE_THRESHOLD_MODEL,
                }
            )
        return [item for item in candidates if item["model_key"]]

    def _sync_model_paper_trades(self, owner_username: str, snapshot: SnapshotComputation) -> None:
        trade_date = snapshot.updated_at.date().isoformat()
        candidates = self._build_paper_trade_candidates(snapshot)
        candidate_keys = {str(item["model_key"]) for item in candidates}
        existing_open_rows = list_model_paper_trades(
            self.settings.db_path,
            owner_username,
            symbol=snapshot.symbol,
            status="open",
        )
        for row in existing_open_rows:
            if str(row["model_key"]) in candidate_keys:
                continue
            entry_price = float(row["entry_price"] or 0.0)
            current_return_pct = round((snapshot.latest_price - entry_price) / entry_price * 100, 2) if entry_price > 0 else 0.0
            entry_date = date.fromisoformat(str(row["entry_date"])[:10])
            holding_days = max(0, (snapshot.updated_at.date() - entry_date).days)
            max_return_pct = max(float(row["max_return_pct"] or 0.0), current_return_pct)
            min_return_pct = min(float(row["min_return_pct"] or 0.0), current_return_pct)
            max_drawdown_pct = max(float(row["max_drawdown_pct"] or 0.0), max_return_pct - current_return_pct)
            close_model_paper_trade(
                self.settings.db_path,
                trade_id=int(row["id"]),
                exit_price=snapshot.latest_price,
                exit_date=trade_date,
                holding_days=holding_days,
                max_return_pct=max_return_pct,
                min_return_pct=min_return_pct,
                max_drawdown_pct=max_drawdown_pct,
                realized_return_pct=current_return_pct,
                exit_reason="模型已移出当前组合，纸面持仓自动结束。",
            )
        for item in candidates:
            score = float(item["score"])
            trade = get_open_model_paper_trade(
                self.settings.db_path,
                owner_username,
                snapshot.symbol,
                str(item["model_key"]),
            )
            if trade is None and score >= float(item["open_threshold"]) and snapshot.latest_price > 0:
                open_model_paper_trade(
                    self.settings.db_path,
                    owner_username=owner_username,
                    symbol=snapshot.symbol,
                    display_name=snapshot.display_name,
                    model_scope=str(item["model_scope"]),
                    model_key=str(item["model_key"]),
                    model_label=str(item["model_label"]),
                    entry_price=snapshot.latest_price,
                    entry_date=trade_date,
                    entry_reason=f"量化评分 {score:.2f} 分，达到入场阈值 {float(item['open_threshold']):.2f}。",
                )
                continue
            if trade is None:
                continue

            entry_price = float(trade["entry_price"] or 0.0)
            if entry_price <= 0:
                continue
            current_return_pct = round((snapshot.latest_price - entry_price) / entry_price * 100, 2)
            max_return_pct = max(float(trade["max_return_pct"] or 0.0), current_return_pct)
            min_return_pct = min(float(trade["min_return_pct"] or 0.0), current_return_pct)
            max_drawdown_pct = max(float(trade["max_drawdown_pct"] or 0.0), max_return_pct - current_return_pct)
            entry_date = date.fromisoformat(str(trade["entry_date"])[:10])
            holding_days = max(0, (snapshot.updated_at.date() - entry_date).days)
            mark_model_paper_trade(
                self.settings.db_path,
                trade_id=int(trade["id"]),
                latest_price=snapshot.latest_price,
                latest_date=trade_date,
                holding_days=holding_days,
                max_return_pct=max_return_pct,
                min_return_pct=min_return_pct,
                max_drawdown_pct=max_drawdown_pct,
                unrealized_return_pct=current_return_pct,
            )
            exit_reason = self._resolve_paper_trade_exit_reason(
                model_scope=str(item["model_scope"]),
                score=score,
                close_threshold=float(item["close_threshold"]),
                holding_days=holding_days,
                current_return_pct=current_return_pct,
            )
            if exit_reason:
                close_model_paper_trade(
                    self.settings.db_path,
                    trade_id=int(trade["id"]),
                    exit_price=snapshot.latest_price,
                    exit_date=trade_date,
                    holding_days=holding_days,
                    max_return_pct=max_return_pct,
                    min_return_pct=min_return_pct,
                    max_drawdown_pct=max_drawdown_pct,
                    realized_return_pct=current_return_pct,
                    exit_reason=exit_reason,
                )

    @staticmethod
    def _resolve_paper_trade_exit_reason(
        model_scope: str,
        score: float,
        close_threshold: float,
        holding_days: int,
        current_return_pct: float,
    ) -> str | None:
        max_holding_days = (
            PAPER_TRADE_MAX_HOLDING_DAYS_GROUP if model_scope == "group" else PAPER_TRADE_MAX_HOLDING_DAYS_MODEL
        )
        if current_return_pct <= PAPER_TRADE_STOP_LOSS_PCT:
            return (
                f"止损退出：当前收益 {current_return_pct:.2f}% ，"
                f"触发止损阈值 {PAPER_TRADE_STOP_LOSS_PCT:.2f}% 。"
            )
        if current_return_pct >= PAPER_TRADE_TAKE_PROFIT_PCT:
            return (
                f"止盈退出：当前收益 {current_return_pct:.2f}% ，"
                f"达到止盈阈值 {PAPER_TRADE_TAKE_PROFIT_PCT:.2f}% 。"
            )
        if holding_days >= max_holding_days:
            return (
                f"超期退出：已持有 {holding_days} 天，"
                f"达到最长持有周期 {max_holding_days} 天。"
            )
        if score <= close_threshold:
            return f"模型走弱退出：量化评分 {score:.2f} 分，跌破离场阈值 {close_threshold:.2f}。"
        return None

    def _process_intraday_signal(
        self,
        owner_username: str,
        symbol: str,
        display_name: str,
        trigger_type: str,
        trade_marker: str,
        condition_met: bool,
        detail: str,
        current_price: float,
        indicator_values: dict[str, float | str],
        email_settings: Any,
        required_hits: int = 2,
    ) -> None:
        state = get_signal_state(self.settings.db_path, owner_username, symbol, trigger_type)
        hits = 1 if condition_met and state is None else 0
        if condition_met and state is not None:
            hits = state["consecutive_hits"] + 1
        if not condition_met:
            hits = 0

        last_event_marker = state["last_event_marker"] if state else None
        should_alert = condition_met and hits >= required_hits and last_event_marker != trade_marker
        if should_alert:
            self._emit_alert(
                owner_username=owner_username,
                symbol=symbol,
                display_name=display_name,
                trigger_type=trigger_type,
                current_price=current_price,
                detail=detail,
                indicator_values=indicator_values,
                email_settings=email_settings,
                triggered_at=datetime.now(UTC).isoformat(),
            )
            last_event_marker = trade_marker

        upsert_signal_state(
            db_path=self.settings.db_path,
            owner_username=owner_username,
            symbol=symbol,
            trigger_type=trigger_type,
            consecutive_hits=hits,
            last_condition_met=condition_met,
            last_event_marker=last_event_marker,
            pending_delivery=bool(state and state["pending_delivery"]),
            deliver_on=state["deliver_on"] if state else None,
            pending_payload=json.loads(state["pending_payload"]) if state and state["pending_payload"] else None,
        )

    def _prepare_dividend_alerts(self, trade_date: date) -> None:
        marker = trade_date.isoformat()
        snapshot_cache: dict[tuple[str, tuple[str, ...], str], SnapshotComputation] = {}
        owner_position_cache: dict[str, dict[str, dict[str, Any]]] = {}
        owner_portfolio_base_cache: dict[str, float] = {}
        owners = sorted({row["owner_username"] for row in get_monitored_stocks(self.settings.db_path)})
        for owner_username in owners:
            quant_config = self._resolve_owner_quant_config(owner_username)
            owner_positions = owner_position_cache.setdefault(owner_username, self._resolve_owner_position_map(owner_username))
            cache_suffix = json.dumps(quant_config["strategy_params"], ensure_ascii=False, sort_keys=True)
            job_state = get_job_state(self.settings.db_path, owner_username, "dividend_prep")
            if job_state and job_state["last_run_marker"] == marker:
                continue
            for stock in get_monitored_stocks(self.settings.db_path, owner_username):
                cache_key = (
                    stock["symbol"],
                    tuple(quant_config["selected_models"]),
                    cache_suffix,
                    json.dumps(quant_config["paper_trade_feedback"], ensure_ascii=False, sort_keys=True),
                )
                snapshot = snapshot_cache.get(cache_key)
                if snapshot is None:
                    snapshot = self.build_snapshot(
                        stock["symbol"],
                        selected_models=quant_config["selected_models"],
                        strategy_params=quant_config["strategy_params"],
                        live_feedback=quant_config["paper_trade_feedback"],
                    )
                    snapshot_cache[cache_key] = snapshot
                stock_advice = build_stock_comprehensive_advice(snapshot.__dict__)
                high_dividend_state = get_signal_state(self.settings.db_path, owner_username, stock["symbol"], "股息率")
                if (
                    snapshot.dividend_yield >= 4.5
                    and self._should_emit_buy_alert(snapshot, "股息率", stock_advice)
                    and (high_dividend_state is None or high_dividend_state["last_event_marker"] != marker)
                ):
                    payload = self._build_pending_payload(
                        owner_username=owner_username,
                        symbol=snapshot.symbol,
                        display_name=snapshot.display_name,
                        trigger_type="股息率",
                        current_price=snapshot.latest_price,
                        detail=snapshot.trigger_detail,
                        indicator_values={
                            "dividend_yield": snapshot.dividend_yield,
                            "latest_price": snapshot.latest_price,
                        },
                        event_marker=marker,
                    )
                    upsert_signal_state(
                        db_path=self.settings.db_path,
                        owner_username=owner_username,
                        symbol=snapshot.symbol,
                        trigger_type="股息率",
                        consecutive_hits=1,
                        last_condition_met=True,
                        last_event_marker=high_dividend_state["last_event_marker"] if high_dividend_state else None,
                        pending_delivery=True,
                        deliver_on=marker,
                        pending_payload=payload,
                    )
                low_dividend_state = get_signal_state(self.settings.db_path, owner_username, stock["symbol"], "低股息率卖出")
                if (
                    0 < snapshot.dividend_yield < SELL_DIVIDEND_THRESHOLD
                    and self._should_emit_sell_alert(owner_positions.get(stock["symbol"]), snapshot, "低股息率卖出", stock_advice)
                    and (low_dividend_state is None or low_dividend_state["last_event_marker"] != marker)
                ):
                    payload = self._build_pending_payload(
                        owner_username=owner_username,
                        symbol=snapshot.symbol,
                        display_name=snapshot.display_name,
                        trigger_type="低股息率卖出",
                        current_price=snapshot.latest_price,
                        detail=f"{snapshot.trigger_detail} | 卖出参考：股息率低于 {SELL_DIVIDEND_THRESHOLD:.2f}% 可视为防守属性减弱。",
                        indicator_values={
                            "dividend_yield": snapshot.dividend_yield,
                            "sell_dividend_threshold": SELL_DIVIDEND_THRESHOLD,
                            "latest_price": snapshot.latest_price,
                        },
                        event_marker=marker,
                    )
                    upsert_signal_state(
                        db_path=self.settings.db_path,
                        owner_username=owner_username,
                        symbol=snapshot.symbol,
                        trigger_type="低股息率卖出",
                        consecutive_hits=1,
                        last_condition_met=True,
                        last_event_marker=low_dividend_state["last_event_marker"] if low_dividend_state else None,
                        pending_delivery=True,
                        deliver_on=marker,
                        pending_payload=payload,
                    )
            set_job_state(self.settings.db_path, owner_username, "dividend_prep", marker)

    def _prepare_weekly_cross_alerts(self, trade_date: date) -> None:
        marker = trade_date.isoformat()
        next_trade_day = self.trade_calendar.next_trading_day(trade_date)
        if next_trade_day is None:
            return
        snapshot_cache: dict[tuple[str, tuple[str, ...], str], SnapshotComputation] = {}
        owner_position_cache: dict[str, dict[str, dict[str, Any]]] = {}
        owner_portfolio_base_cache: dict[str, float] = {}
        owners = sorted({row["owner_username"] for row in get_monitored_stocks(self.settings.db_path)})
        for owner_username in owners:
            quant_config = self._resolve_owner_quant_config(owner_username)
            owner_positions = owner_position_cache.setdefault(owner_username, self._resolve_owner_position_map(owner_username))
            owner_total_investment = owner_portfolio_base_cache.setdefault(
                owner_username,
                float(get_portfolio_settings(self.settings.db_path, owner_username)["total_investment_amount"] or 0.0),
            )
            cache_suffix = json.dumps(quant_config["strategy_params"], ensure_ascii=False, sort_keys=True)
            job_state = get_job_state(self.settings.db_path, owner_username, "weekly_cross_prep")
            if job_state and job_state["last_run_marker"] == marker:
                continue
            for stock in get_monitored_stocks(self.settings.db_path, owner_username):
                cache_key = (
                    stock["symbol"],
                    tuple(quant_config["selected_models"]),
                    cache_suffix,
                    json.dumps(quant_config["paper_trade_feedback"], ensure_ascii=False, sort_keys=True),
                )
                snapshot = snapshot_cache.get(cache_key)
                if snapshot is None:
                    snapshot = self.build_snapshot(
                        stock["symbol"],
                        selected_models=quant_config["selected_models"],
                        strategy_params=quant_config["strategy_params"],
                        live_feedback=quant_config["paper_trade_feedback"],
                    )
                    snapshot_cache[cache_key] = snapshot
                position_summary = owner_positions.get(stock["symbol"])
                position_concentration_pct = self._estimate_position_concentration_pct(
                    stock["symbol"],
                    snapshot.latest_price,
                    position_summary,
                    owner_positions,
                    owner_total_investment,
                )
                snapshot = replace(snapshot, position_concentration_pct=position_concentration_pct)
                stock_advice = build_stock_comprehensive_advice(snapshot.__dict__)
                stock_advice["position_concentration_pct"] = position_concentration_pct
                deliver_on = next_trade_day.isoformat()
                self._queue_weekly_pending_alert(
                    owner_username=owner_username,
                    snapshot=snapshot,
                    trigger_type="30周线上穿60周线",
                    marker=marker,
                    deliver_on=deliver_on,
                    condition_met=snapshot.weekly_bullish_crossed,
                    indicator_values={"ma_30w": snapshot.ma_30w, "ma_60w": snapshot.ma_60w},
                    detail=f"{snapshot.trigger_detail} | 周线信号：30周线本周上穿60周线。",
                )
                self._queue_weekly_pending_alert(
                    owner_username=owner_username,
                    snapshot=snapshot,
                    trigger_type="30周线下穿60周线",
                    marker=marker,
                    deliver_on=deliver_on,
                    condition_met=snapshot.weekly_bearish_crossed,
                    indicator_values={"ma_30w": snapshot.ma_30w, "ma_60w": snapshot.ma_60w},
                    detail=f"{snapshot.trigger_detail} | 周线信号：30周线本周下穿60周线。",
                )
                self._queue_weekly_pending_alert(
                    owner_username=owner_username,
                    snapshot=snapshot,
                    trigger_type="有效突破60周线",
                    marker=marker,
                    deliver_on=deliver_on,
                    condition_met=snapshot.weekly_breakout_above_ma60w,
                    indicator_values={
                        "weekly_close": snapshot.weekly_close,
                        "ma_60w": snapshot.ma_60w,
                        "breakout_close_ratio": WEEKLY_BREAKOUT_MIN_CLOSE_RATIO,
                    },
                    detail=(
                        f"{snapshot.trigger_detail} | 周线信号：周收盘有效突破60周线，"
                        "且本周低点基本站稳在60周线上方。"
                    ),
                )
            set_job_state(self.settings.db_path, owner_username, "weekly_cross_prep", marker)

    def _queue_weekly_pending_alert(
        self,
        owner_username: str,
        snapshot: SnapshotComputation,
        trigger_type: str,
        marker: str,
        deliver_on: str,
        condition_met: bool,
        indicator_values: dict[str, float | str],
        detail: str,
    ) -> None:
        if not condition_met:
            return
        state = get_signal_state(self.settings.db_path, owner_username, snapshot.symbol, trigger_type)
        if state is not None and state["last_event_marker"] == marker:
            return
        payload = self._build_pending_payload(
            owner_username=owner_username,
            symbol=snapshot.symbol,
            display_name=snapshot.display_name,
            trigger_type=trigger_type,
            current_price=snapshot.latest_price,
            detail=detail,
            indicator_values=indicator_values,
            event_marker=marker,
        )
        upsert_signal_state(
            db_path=self.settings.db_path,
            owner_username=owner_username,
            symbol=snapshot.symbol,
            trigger_type=trigger_type,
            consecutive_hits=1,
            last_condition_met=True,
            last_event_marker=state["last_event_marker"] if state else None,
            pending_delivery=True,
            deliver_on=deliver_on,
            pending_payload=payload,
        )

    def _deliver_pending_alerts(self, trade_date: date) -> None:
        for state in list_pending_signal_states(self.settings.db_path, trade_date.isoformat()):
            payload = json.loads(state["pending_payload"] or "{}")
            if not payload:
                continue
            email_settings = get_email_settings(self.settings.db_path, state["owner_username"])
            triggered_at = datetime.now(UTC).isoformat()
            self._emit_alert(
                owner_username=state["owner_username"],
                symbol=payload["symbol"],
                display_name=payload["display_name"],
                trigger_type=payload["trigger_type"],
                current_price=float(payload["current_price"]),
                detail=str(payload["detail"]),
                indicator_values=payload["indicator_values"],
                email_settings=email_settings,
                triggered_at=triggered_at,
            )
            upsert_signal_state(
                db_path=self.settings.db_path,
                owner_username=state["owner_username"],
                symbol=state["symbol"],
                trigger_type=state["trigger_type"],
                consecutive_hits=state["consecutive_hits"],
                last_condition_met=bool(state["last_condition_met"]),
                last_event_marker=payload.get("event_marker"),
                pending_delivery=False,
                deliver_on=None,
                pending_payload=None,
            )

    def _emit_alert(
        self,
        owner_username: str,
        symbol: str,
        display_name: str,
        trigger_type: str,
        current_price: float,
        detail: str,
        indicator_values: dict[str, float | str],
        email_settings: Any,
        triggered_at: str,
    ) -> None:
        payload = {
            "symbol": symbol,
            "display_name": display_name,
            "trigger_type": trigger_type,
            "current_price": current_price,
            "detail": detail,
            "indicator_values": indicator_values,
            "triggered_at": triggered_at,
        }
        snapshot_row = get_snapshot(self.settings.db_path, owner_username, symbol)
        if snapshot_row is not None:
            snapshot_payload = dict(snapshot_row)
            stock_advice = build_stock_comprehensive_advice(snapshot_payload)
            payload.update(
                {
                    "market_environment": stock_advice.get("market_environment"),
                    "market_bias_score": stock_advice.get("market_bias_score"),
                    "industry_name": snapshot_payload.get("industry_name"),
                    "industry_environment": stock_advice.get("industry_environment"),
                    "latest_volume_ratio": stock_advice.get("latest_volume_ratio"),
                    "earnings_phase": stock_advice.get("earnings_phase"),
                    "action": stock_advice.get("action"),
                    "action_reason": stock_advice.get("action_reason"),
                    "decision_summary": stock_advice.get("decision_summary"),
                    "decision_reason_lines": stock_advice.get("decision_reason_lines"),
                    "buy_signal_summary": stock_advice.get("buy_signal_summary"),
                    "sell_signal_summary": stock_advice.get("sell_signal_summary"),
                    "trigger_interpretation": self._build_trigger_interpretation(trigger_type, stock_advice),
                }
            )
        email_result = send_message(
            email_settings,
            subject=f"[{trigger_type}] {symbol} {display_name}",
            body=build_alert_email_body(payload),
            html_body=build_alert_email_html_body(payload),
        )
        add_alert_history(
            db_path=self.settings.db_path,
            owner_username=owner_username,
            symbol=symbol,
            display_name=display_name,
            trigger_type=trigger_type,
            current_price=current_price,
            indicator_values=indicator_values,
            email_status=email_result.status,
            email_error=email_result.error,
            triggered_at=triggered_at,
        )

    @staticmethod
    def _build_trigger_interpretation(trigger_type: str, stock_advice: dict[str, Any]) -> str:
        action = str(stock_advice.get("action") or "观望")
        trigger_is_sell = "卖出" in trigger_type or trigger_type == "30周线下穿60周线"
        trigger_is_buy = not trigger_is_sell
        if trigger_is_sell and action == "偏买入":
            return "这是一条局部止盈或风控提醒：虽然整体买入信号仍有优势，但当前价格/仓位已触发减仓条件，更适合先做分批兑现。"
        if trigger_is_buy and action == "偏卖出":
            return "这是一条局部支撑或回踩提醒：它说明出现了可观察的买点线索，但整体仓位建议仍偏谨慎，不等于可以立即重仓抄底。"
        if action == "观望":
            return "这次提醒更适合作为观察信号使用，需要结合量能、环境和关键价位确认后再执行。"
        return "这次提醒与当前总体结论方向基本一致，可按计划分批执行，不建议情绪化一次性操作。"

    @staticmethod
    def _build_pending_payload(
        owner_username: str,
        symbol: str,
        display_name: str,
        trigger_type: str,
        current_price: float,
        detail: str,
        indicator_values: dict[str, float | str],
        event_marker: str,
    ) -> dict[str, Any]:
        return {
            "owner_username": owner_username,
            "symbol": symbol,
            "display_name": display_name,
            "trigger_type": trigger_type,
            "current_price": current_price,
            "detail": detail,
            "indicator_values": indicator_values,
            "event_marker": event_marker,
        }
