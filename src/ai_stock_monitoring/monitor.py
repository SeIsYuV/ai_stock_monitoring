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

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import math
import re
import threading
from typing import Any

SELL_DIVIDEND_THRESHOLD = 3.5
QUANT_SELL_PROBABILITY_THRESHOLD = 35.0
BUY_TRIGGER_TYPES = {"250日线", "BOLL中轨", "BOLL下轨", "股息率", "30周/60周均线", "量化盈利概率"}
SELL_TRIGGER_TYPES = {"BOLL上轨卖出", "低股息率卖出", "量化走弱卖出"}

from .config import AppSettings
from .database import (
    add_alert_history,
    get_email_settings,
    get_job_state,
    get_monitored_stocks,
    get_portfolio_settings,
    get_quant_settings,
    get_signal_state,
    list_pending_signal_states,
    list_trade_records,
    list_trade_records_for_symbol,
    set_job_state,
    upsert_signal_state,
    upsert_snapshot,
)
from .mailer import build_alert_email_body, build_portfolio_review_email_body, send_message
from .market_hours import MarketStatus, TradeCalendar, get_market_status
from .providers import load_provider
from .providers.base import PriceBar, Quote
from .quant import DEFAULT_QUANT_MODELS, build_quant_signal, normalize_selected_models, normalize_strategy_params
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
    if not all([prev_ma_30w, prev_ma_60w, current_ma_30w, current_ma_60w]):
        return False
    prev_diff = prev_ma_30w - prev_ma_60w
    current_diff = current_ma_30w - current_ma_60w
    return prev_diff == 0 or current_diff == 0 or prev_diff * current_diff < 0


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

    ma_250 = calculate_simple_moving_average(daily_closes, 250)
    boll_mid = calculate_simple_moving_average(daily_closes, 20)
    boll_lower = calculate_bollinger_lower_band(daily_closes, 20)
    boll_upper = calculate_bollinger_upper_band(daily_closes, 20)
    ma_30w = calculate_simple_moving_average(weekly_closes, 30)
    ma_60w = calculate_simple_moving_average(weekly_closes, 60)
    prev_ma_30w = calculate_simple_moving_average(weekly_closes[:-1], 30)
    prev_ma_60w = calculate_simple_moving_average(weekly_closes[:-1], 60)
    weekly_crossed = has_weekly_crossed(prev_ma_30w, prev_ma_60w, ma_30w, ma_60w)

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
    if weekly_crossed:
        triggered_labels.append("30周/60周均线")

    trigger_state = "、".join(triggered_labels) if triggered_labels else "正常"
    trigger_detail = (
        f"现价 {quote.latest_price:.2f} | 250日线 {ma_250:.2f} | "
        f"30周/60周 {ma_30w:.2f}/{ma_60w:.2f} | "
        f"BOLL上/中/下轨 {boll_upper:.2f}/{boll_mid:.2f}/{boll_lower:.2f} | 股息率 {dividend_yield:.2f}% | "
        f"量化综合盈利概率 {quant_probability:.2f}%"
    )
    return SnapshotComputation(
        symbol=symbol,
        display_name=quote.name,
        latest_price=quote.latest_price,
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
    ) -> SnapshotComputation:
        """Build a snapshot for one symbol.

        盘中优先使用实时价格；如果盘后或接口短暂返回 0，
        就退回到最近一个日线收盘价，这样新增股票后在非交易时段也能立刻看到最近可用数据。
        """

        quote = self.provider.get_quote(symbol)
        daily_bars = self.provider.get_daily_bars(symbol)
        weekly_bars = self.provider.get_weekly_bars(symbol)

        effective_quote = quote
        if quote.latest_price <= 0 and daily_bars:
            latest_daily_bar = daily_bars[-1]
            effective_quote = Quote(
                symbol=quote.symbol,
                name=quote.name,
                latest_price=latest_daily_bar.close_price,
                updated_at=datetime.combine(
                    latest_daily_bar.traded_on,
                    datetime.min.time(),
                    tzinfo=quote.updated_at.tzinfo,
                ),
            )

        dividend_yield = self.provider.get_trailing_dividend_yield(symbol, effective_quote.latest_price)
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
            selected_models=normalize_selected_models(selected_models),
            strategy_params=normalize_strategy_params(strategy_params),
        )
        return compute_snapshot_metrics(
            symbol=symbol,
            quote=effective_quote,
            daily_bars=daily_bars,
            weekly_bars=weekly_bars,
            dividend_yield=dividend_yield,
            quant_probability=quant_signal.probability,
            quant_model_breakdown=quant_signal.breakdown_json,
        )

    def refresh_symbol_snapshot(self, owner_username: str, symbol: str) -> SnapshotComputation:
        quant_config = self._resolve_owner_quant_config(owner_username)
        snapshot = self.build_snapshot(
            symbol,
            selected_models=quant_config["selected_models"],
            strategy_params=quant_config["strategy_params"],
        )
        upsert_snapshot(
            db_path=self.settings.db_path,
            owner_username=owner_username,
            symbol=snapshot.symbol,
            display_name=snapshot.display_name,
            latest_price=snapshot.latest_price,
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
            updated_at=snapshot.updated_at.isoformat(),
        )
        self.last_refresh_at = datetime.now(UTC)
        return snapshot

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
                selected_models = quant_settings["selected_models"]
                strategy_params = quant_settings["strategy_params"]
                cache_key = (
                    symbol,
                    tuple(selected_models),
                    json.dumps(strategy_params, ensure_ascii=False, sort_keys=True),
                )
                snapshot = snapshot_cache.get(cache_key)
                if snapshot is None:
                    snapshot = self.build_snapshot(
                        symbol,
                        selected_models=selected_models,
                        strategy_params=strategy_params,
                    )
                    snapshot_cache[cache_key] = snapshot
                upsert_snapshot(
                    db_path=self.settings.db_path,
                    owner_username=owner_username,
                    symbol=snapshot.symbol,
                    display_name=snapshot.display_name,
                    latest_price=snapshot.latest_price,
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
                    updated_at=snapshot.updated_at.isoformat(),
                )
                stock_advice = build_stock_comprehensive_advice(snapshot.__dict__)
                buy_alert_enabled = stock_advice["action"] == "偏买入"
                position_summary = owner_positions.get(symbol)
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="250日线",
                    trade_marker=trade_marker,
                    condition_met=bool(snapshot.ma_250 and snapshot.latest_price <= snapshot.ma_250 and self._should_emit_buy_alert(snapshot, "250日线", stock_advice["action"])),
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
                    condition_met=bool(snapshot.boll_mid and snapshot.latest_price <= snapshot.boll_mid and self._should_emit_buy_alert(snapshot, "BOLL中轨", stock_advice["action"])),
                    detail=snapshot.trigger_detail,
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
                    condition_met=bool(snapshot.boll_lower and snapshot.latest_price <= snapshot.boll_lower and self._should_emit_buy_alert(snapshot, "BOLL下轨", stock_advice["action"])),
                    detail=snapshot.trigger_detail,
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
                    condition_met=bool(snapshot.boll_upper and snapshot.latest_price >= snapshot.boll_upper and self._should_emit_sell_alert(position_summary, "BOLL上轨卖出")),
                    detail=f"{snapshot.trigger_detail} | 卖出参考：价格突破 BOLL 上轨，可结合仓位分批止盈。",
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
                        condition_met=bool(snapshot.quant_probability >= threshold and buy_alert_enabled and self._should_emit_buy_alert(snapshot, "量化盈利概率", stock_advice["action"])),
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
                        condition_met=bool(quant_sell_signal.should_alert and self._should_emit_sell_alert(position_summary, "量化走弱卖出")),
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
    def _should_emit_buy_alert(snapshot: SnapshotComputation, trigger_type: str, action: str) -> bool:
        if trigger_type not in BUY_TRIGGER_TYPES:
            return False
        if action != "偏买入":
            return False
        if trigger_type == "量化盈利概率":
            return any(label in BUY_TRIGGER_TYPES - {"量化盈利概率"} for label in snapshot.triggered_labels)
        return trigger_type in snapshot.triggered_labels

    @staticmethod
    def _should_emit_sell_alert(position_summary: dict[str, Any] | None, trigger_type: str) -> bool:
        if trigger_type not in SELL_TRIGGER_TYPES:
            return False
        if not position_summary:
            return False
        return int(position_summary.get("position_quantity") or 0) > 0

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
            quant_config = self._resolve_owner_quant_config(owner_username)
            snapshots: list[dict[str, Any]] = []
            for symbol in sorted(position_map):
                snapshot = self.build_snapshot(
                    symbol,
                    selected_models=quant_config["selected_models"],
                    strategy_params=quant_config["strategy_params"],
                )
                upsert_snapshot(
                    db_path=self.settings.db_path,
                    owner_username=owner_username,
                    symbol=snapshot.symbol,
                    display_name=snapshot.display_name,
                    latest_price=snapshot.latest_price,
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
                    updated_at=snapshot.updated_at.isoformat(),
                )
                snapshots.append({
                    "symbol": snapshot.symbol,
                    "display_name": snapshot.display_name,
                    "latest_price": snapshot.latest_price,
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
                    "updated_at": snapshot.updated_at.isoformat(),
                })
            portfolio_settings = get_portfolio_settings(self.settings.db_path, owner_username)
            portfolio_profile = build_portfolio_profile(
                list_trade_records(self.settings.db_path, owner_username, None),
                snapshots,
                float(portfolio_settings["total_investment_amount"] or 0.0),
            )
            email_result = send_message(
                email_settings,
                subject=f"[收盘持仓复盘] {owner_username} {marker}",
                body=build_portfolio_review_email_body(
                    {
                        "owner_username": owner_username,
                        "trade_date": marker,
                        "portfolio_profile": portfolio_profile,
                    }
                ),
            )
            if not email_result.success:
                self.last_error_message = f"{owner_username} 收盘持仓复盘邮件发送失败：{email_result.error}"
            set_job_state(self.settings.db_path, owner_username, "post_close_holding_review", marker)

    def _resolve_owner_quant_config(self, owner_username: str) -> dict[str, Any]:
        """Load one user's quant configuration and normalize it for downstream use."""

        settings = get_quant_settings(self.settings.db_path, owner_username)
        return {
            "enabled": bool(settings["enabled"]),
            "probability_threshold": float(settings["probability_threshold"]),
            "selected_models": tuple(json.loads(settings["selected_models"] or "[]")) or DEFAULT_QUANT_MODELS,
            "strategy_params": normalize_strategy_params(json.loads(settings["strategy_params"] or "{}")),
        }

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
        owners = sorted({row["owner_username"] for row in get_monitored_stocks(self.settings.db_path)})
        for owner_username in owners:
            quant_config = self._resolve_owner_quant_config(owner_username)
            owner_positions = owner_position_cache.setdefault(owner_username, self._resolve_owner_position_map(owner_username))
            cache_suffix = json.dumps(quant_config["strategy_params"], ensure_ascii=False, sort_keys=True)
            job_state = get_job_state(self.settings.db_path, owner_username, "dividend_prep")
            if job_state and job_state["last_run_marker"] == marker:
                continue
            for stock in get_monitored_stocks(self.settings.db_path, owner_username):
                cache_key = (stock["symbol"], tuple(quant_config["selected_models"]), cache_suffix)
                snapshot = snapshot_cache.get(cache_key)
                if snapshot is None:
                    snapshot = self.build_snapshot(
                        stock["symbol"],
                        selected_models=quant_config["selected_models"],
                        strategy_params=quant_config["strategy_params"],
                    )
                    snapshot_cache[cache_key] = snapshot
                stock_advice = build_stock_comprehensive_advice(snapshot.__dict__)
                high_dividend_state = get_signal_state(self.settings.db_path, owner_username, stock["symbol"], "股息率")
                if (
                    snapshot.dividend_yield >= 4.5
                    and self._should_emit_buy_alert(snapshot, "股息率", stock_advice["action"])
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
                    and self._should_emit_sell_alert(owner_positions.get(stock["symbol"]), "低股息率卖出")
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
        owners = sorted({row["owner_username"] for row in get_monitored_stocks(self.settings.db_path)})
        for owner_username in owners:
            quant_config = self._resolve_owner_quant_config(owner_username)
            cache_suffix = json.dumps(quant_config["strategy_params"], ensure_ascii=False, sort_keys=True)
            job_state = get_job_state(self.settings.db_path, owner_username, "weekly_cross_prep")
            if job_state and job_state["last_run_marker"] == marker:
                continue
            for stock in get_monitored_stocks(self.settings.db_path, owner_username):
                cache_key = (stock["symbol"], tuple(quant_config["selected_models"]), cache_suffix)
                snapshot = snapshot_cache.get(cache_key)
                if snapshot is None:
                    snapshot = self.build_snapshot(
                        stock["symbol"],
                        selected_models=quant_config["selected_models"],
                        strategy_params=quant_config["strategy_params"],
                    )
                    snapshot_cache[cache_key] = snapshot
                stock_advice = build_stock_comprehensive_advice(snapshot.__dict__)
                state = get_signal_state(self.settings.db_path, owner_username, stock["symbol"], "30周/60周均线")
                if (
                    snapshot.weekly_crossed
                    and self._should_emit_buy_alert(snapshot, "30周/60周均线", stock_advice["action"])
                    and (state is None or state["last_event_marker"] != marker)
                ):
                    payload = self._build_pending_payload(
                        owner_username=owner_username,
                        symbol=snapshot.symbol,
                        display_name=snapshot.display_name,
                        trigger_type="30周/60周均线",
                        current_price=snapshot.latest_price,
                        detail=snapshot.trigger_detail,
                        indicator_values={
                            "ma_30w": snapshot.ma_30w,
                            "ma_60w": snapshot.ma_60w,
                        },
                        event_marker=marker,
                    )
                    upsert_signal_state(
                        db_path=self.settings.db_path,
                        owner_username=owner_username,
                        symbol=snapshot.symbol,
                        trigger_type="30周/60周均线",
                        consecutive_hits=1,
                        last_condition_met=True,
                        last_event_marker=state["last_event_marker"] if state else None,
                        pending_delivery=True,
                        deliver_on=next_trade_day.isoformat(),
                        pending_payload=payload,
                    )
            set_job_state(self.settings.db_path, owner_username, "weekly_cross_prep", marker)

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
        email_result = send_message(
            email_settings,
            subject=f"[{trigger_type}] {symbol} {display_name}",
            body=build_alert_email_body(payload),
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
