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

from .config import AppSettings
from .database import (
    add_alert_history,
    get_email_settings,
    get_job_state,
    get_monitored_stocks,
    get_quant_settings,
    get_signal_state,
    list_pending_signal_states,
    set_job_state,
    upsert_signal_state,
    upsert_snapshot,
)
from .mailer import build_alert_email_body, send_message
from .market_hours import MarketStatus, TradeCalendar, get_market_status
from .providers import load_provider
from .providers.base import PriceBar, Quote
from .quant import DEFAULT_QUANT_MODELS, build_quant_signal, normalize_selected_models, normalize_strategy_params


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
    if dividend_yield >= 4.5:
        triggered_labels.append("股息率")
    if weekly_crossed:
        triggered_labels.append("30周/60周均线")

    trigger_state = "、".join(triggered_labels) if triggered_labels else "正常"
    trigger_detail = (
        f"现价 {quote.latest_price:.2f} | 250日线 {ma_250:.2f} | "
        f"30周/60周 {ma_30w:.2f}/{ma_60w:.2f} | "
        f"BOLL中轨/下轨 {boll_mid:.2f}/{boll_lower:.2f} | 股息率 {dividend_yield:.2f}% | "
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

    def build_chart_payload(self, symbol: str) -> dict[str, list[float | str | None]]:
        daily_bars = self.provider.get_daily_bars(symbol, limit=max(self.settings.detail_chart_days, 20))
        closes = [float(item.close_price) for item in daily_bars]
        labels = [item.traded_on.isoformat() for item in daily_bars[-self.settings.detail_chart_days :]]
        close_series = closes[-self.settings.detail_chart_days :]
        ma_250_series: list[float | None] = []
        boll_mid_series: list[float | None] = []
        boll_lower_series: list[float | None] = []
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
        return {
            "labels": labels,
            "close": close_series,
            "ma250": ma_250_series[-self.settings.detail_chart_days :],
            "bollMid": boll_mid_series[-self.settings.detail_chart_days :],
            "bollLower": boll_lower_series[-self.settings.detail_chart_days :],
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
            if status.is_post_close and self.trade_calendar.is_last_trading_day_of_week(trade_date):
                self._prepare_weekly_cross_alerts(trade_date)
        except Exception as exc:  # pragma: no cover
            self.last_error_message = str(exc)
        finally:
            self._lock.release()

    def _refresh_open_session(self, trade_date: date) -> None:
        snapshot_cache: dict[tuple[str, tuple[str, ...], str], SnapshotComputation] = {}
        for stock in get_monitored_stocks(self.settings.db_path):
            owner_username = stock["owner_username"]
            symbol = stock["symbol"]
            try:
                email_settings = get_email_settings(self.settings.db_path, owner_username)
                quant_settings = self._resolve_owner_quant_config(owner_username)
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
                    dividend_yield=snapshot.dividend_yield,
                    quant_probability=snapshot.quant_probability,
                    quant_model_breakdown=snapshot.quant_model_breakdown,
                    trigger_state=snapshot.trigger_state,
                    trigger_detail=snapshot.trigger_detail,
                    updated_at=snapshot.updated_at.isoformat(),
                )
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="250日线",
                    trade_marker=trade_date.isoformat(),
                    condition_met=bool(snapshot.ma_250 and snapshot.latest_price <= snapshot.ma_250),
                    detail=snapshot.trigger_detail,
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "ma_250": snapshot.ma_250,
                        "latest_price": snapshot.latest_price,
                    },
                    email_settings=email_settings,
                    required_hits=2,
                )
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="BOLL中轨",
                    trade_marker=trade_date.isoformat(),
                    condition_met=bool(snapshot.boll_mid and snapshot.latest_price <= snapshot.boll_mid),
                    detail=snapshot.trigger_detail,
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "boll_mid": snapshot.boll_mid,
                        "latest_price": snapshot.latest_price,
                    },
                    email_settings=email_settings,
                    required_hits=2,
                )
                self._process_intraday_signal(
                    owner_username=owner_username,
                    symbol=symbol,
                    display_name=snapshot.display_name,
                    trigger_type="BOLL下轨",
                    trade_marker=trade_date.isoformat(),
                    condition_met=bool(snapshot.boll_lower and snapshot.latest_price <= snapshot.boll_lower),
                    detail=snapshot.trigger_detail,
                    current_price=snapshot.latest_price,
                    indicator_values={
                        "boll_lower": snapshot.boll_lower,
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
                        trade_marker=trade_date.isoformat(),
                        condition_met=bool(snapshot.quant_probability >= threshold),
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
                self.last_refresh_at = datetime.now(UTC)
                self.last_error_message = None
            except Exception as exc:
                self.last_error_message = f"{owner_username}/{symbol} 刷新失败：{exc}"

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
        owners = sorted({row["owner_username"] for row in get_monitored_stocks(self.settings.db_path)})
        for owner_username in owners:
            quant_config = self._resolve_owner_quant_config(owner_username)
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
                state = get_signal_state(self.settings.db_path, owner_username, stock["symbol"], "股息率")
                if snapshot.dividend_yield >= 4.5 and (state is None or state["last_event_marker"] != marker):
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
                        last_event_marker=state["last_event_marker"] if state else None,
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
                state = get_signal_state(self.settings.db_path, owner_username, stock["symbol"], "30周/60周均线")
                if snapshot.weekly_crossed and (state is None or state["last_event_marker"] != marker):
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
