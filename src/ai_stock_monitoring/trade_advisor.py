from __future__ import annotations

"""Trade journal analysis helpers and LLM integration.

这个模块回答两个问题：
1. 用户现在到底持有多少仓位、成本是多少
2. 当前买卖操作放到趋势和均线里看，是否算合理

如果配置了 `OPENAI_API_KEY`，会优先调用大模型；
否则退回到本地规则分析，保证功能始终可用。
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import requests

from .config import AppSettings


BUY_SIGNAL_WEIGHTS: dict[str, int] = {
    "250日线": 2,
    "BOLL中轨": 1,
    "BOLL下轨": 2,
    "股息率": 2,
    "30周/60周均线": 3,
    "30周线上穿60周线": 3,
    "有效突破60周线": 3,
    "量化盈利概率": 4,
    "择时量化": 4,
}

SELL_SIGNAL_WEIGHTS: dict[str, int] = {
    "BOLL上轨卖出": 2,
    "30周线下穿60周线": 3,
    "低股息率卖出": 2,
    "量化走弱卖出": 4,
    "风险控制卖出": 4,
}

BUY_SIGNAL_SET = set(BUY_SIGNAL_WEIGHTS)
SELL_SIGNAL_SET = set(SELL_SIGNAL_WEIGHTS)

SIGNAL_CATEGORY_LABELS: dict[str, str] = {
    "technical": "技术面",
    "dividend": "红利面",
    "quant": "量化面",
}

SIGNAL_CATEGORY_MAPPING: dict[str, str] = {
    "250日线": "technical",
    "BOLL中轨": "technical",
    "BOLL下轨": "technical",
    "BOLL上轨卖出": "technical",
    "30周/60周均线": "technical",
    "30周线上穿60周线": "technical",
    "30周线下穿60周线": "technical",
    "有效突破60周线": "technical",
    "股息率": "dividend",
    "低股息率卖出": "dividend",
    "量化盈利概率": "quant",
    "量化走弱卖出": "quant",
    "择时量化": "quant",
    "风险控制卖出": "quant",
}


def split_trigger_signals(trigger_state: str | None) -> list[str]:
    if not trigger_state or trigger_state == "正常":
        return []
    return [item for item in str(trigger_state).split("、") if item]


def _build_signal_groups(buy_signals: Sequence[str], sell_signals: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in SIGNAL_CATEGORY_LABELS}
    for signal in buy_signals:
        category = SIGNAL_CATEGORY_MAPPING.get(signal, "technical")
        grouped.setdefault(category, []).append({"side": "buy", "label": signal})
    for signal in sell_signals:
        category = SIGNAL_CATEGORY_MAPPING.get(signal, "technical")
        grouped.setdefault(category, []).append({"side": "sell", "label": signal})
    return [
        {"key": key, "title": SIGNAL_CATEGORY_LABELS[key], "items": grouped[key]}
        for key in ("technical", "dividend", "quant")
        if grouped.get(key)
    ]


def build_market_action_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Blend buy/sell rules and quant strength into one final action suggestion."""

    trigger_signals = split_trigger_signals(snapshot.get("trigger_state"))
    buy_signals = [item for item in trigger_signals if item in BUY_SIGNAL_SET]
    sell_signals = [item for item in trigger_signals if item in SELL_SIGNAL_SET]

    buy_score = sum(BUY_SIGNAL_WEIGHTS.get(item, 0) for item in buy_signals)
    sell_score = sum(SELL_SIGNAL_WEIGHTS.get(item, 0) for item in sell_signals)

    display_buy_signals = list(buy_signals)
    display_sell_signals = list(sell_signals)
    quant_probability = float(snapshot.get("quant_probability") or 0.0)
    market_environment = str(snapshot.get("market_environment") or "中性")
    market_bias_score = float(snapshot.get("market_bias_score") or 0.0)
    industry_environment = str(snapshot.get("industry_environment") or "中性")
    industry_bias_score = float(snapshot.get("industry_bias_score") or 0.0)
    latest_volume_ratio = float(snapshot.get("latest_volume_ratio") or 1.0)
    earnings_phase = str(snapshot.get("earnings_phase") or "常规窗口")
    if quant_probability >= 85:
        buy_score += 3
        if "量化盈利概率" not in display_buy_signals:
            display_buy_signals.append("量化盈利概率")
    elif quant_probability >= 70:
        buy_score += 1
        if "量化盈利概率" not in display_buy_signals:
            display_buy_signals.append("量化盈利概率")
    elif quant_probability <= 35:
        sell_score += 4
        if "量化走弱卖出" not in display_sell_signals:
            display_sell_signals.append("量化走弱卖出")
    elif quant_probability <= 45:
        sell_score += 2
        if "量化走弱卖出" not in display_sell_signals:
            display_sell_signals.append("量化走弱卖出")

    if market_environment == "偏强":
        buy_score += 1
    elif market_environment == "偏弱":
        sell_score += 2
        if "大盘环境偏弱" not in display_sell_signals:
            display_sell_signals.append("大盘环境偏弱")

    if industry_environment == "偏强":
        buy_score += 1
    elif industry_environment == "偏弱":
        sell_score += 1
        if "行业强弱偏弱" not in display_sell_signals:
            display_sell_signals.append("行业强弱偏弱")

    if latest_volume_ratio >= 1.15 and buy_signals:
        buy_score += 1
        if "成交量放大确认" not in display_buy_signals:
            display_buy_signals.append("成交量放大确认")
    elif latest_volume_ratio < 0.8 and buy_signals:
        sell_score += 1
        if "成交量不足" not in display_sell_signals:
            display_sell_signals.append("成交量不足")

    if earnings_phase != "常规窗口":
        sell_score += 1
        if "财报窗口风险" not in display_sell_signals:
            display_sell_signals.append("财报窗口风险")

    dominant_model_label = "量化模型"
    dominant_model_score = 0.0
    try:
        model_breakdown = json.loads(str(snapshot.get("quant_model_breakdown") or "[]"))
    except json.JSONDecodeError:
        model_breakdown = []
    if model_breakdown:
        top_model = max(model_breakdown, key=lambda item: float(item.get("score") or 0.0))
        dominant_model_label = str(top_model.get("label") or dominant_model_label)
        dominant_model_score = float(top_model.get("score") or 0.0)

    score_gap = buy_score - sell_score
    if score_gap >= 3:
        action = "偏买入"
        action_color = "buy"
        action_reason = (
            f"买入信号更强（买入 {buy_score} 分 / 卖出 {sell_score} 分），"
            f"当前最强量化模型为 {dominant_model_label}（{dominant_model_score:.2f} 分）。"
        )
    elif score_gap <= -3:
        action = "偏卖出"
        action_color = "sell"
        action_reason = (
            f"卖出信号更强（卖出 {sell_score} 分 / 买入 {buy_score} 分），"
            f"当前最强量化模型为 {dominant_model_label}（{dominant_model_score:.2f} 分）。"
        )
    else:
        action = "观望"
        action_color = "neutral"
        action_reason = (
            f"买卖信号接近（买入 {buy_score} 分 / 卖出 {sell_score} 分），"
            f"当前最强量化模型为 {dominant_model_label}（{dominant_model_score:.2f} 分），建议等待进一步确认。"
        )

    return {
        "buy_signals": display_buy_signals,
        "sell_signals": display_sell_signals,
        "signal_groups": _build_signal_groups(display_buy_signals, display_sell_signals),
        "action": action,
        "action_color": action_color,
        "action_reason": action_reason,
        "dominant_model_label": dominant_model_label,
        "dominant_model_score": round(dominant_model_score, 2),
        "buy_score": buy_score,
        "sell_score": sell_score,
        "market_environment": market_environment,
        "market_bias_score": round(market_bias_score, 2),
        "industry_environment": industry_environment,
        "industry_bias_score": round(industry_bias_score, 2),
        "latest_volume_ratio": round(latest_volume_ratio, 2),
        "earnings_phase": earnings_phase,
    }



def _clamp_level(raw_score: float) -> int:
    bounded = max(1.0, min(10.0, raw_score))
    return int(math.floor(bounded + 0.5))



def _level_label(level: int) -> str:
    if level >= 9:
        return "极强"
    if level >= 7:
        return "较强"
    if level >= 5:
        return "中性"
    if level >= 3:
        return "较弱"
    return "极弱"



def _build_recommendation_levels(
    snapshot: Mapping[str, Any],
    action_summary: Mapping[str, Any],
) -> dict[str, Any]:
    quant_probability = float(snapshot.get("quant_probability") or 0.0)
    latest_price = float(snapshot.get("latest_price") or 0.0)
    ma_250 = float(snapshot.get("ma_250") or 0.0)
    boll_mid = float(snapshot.get("boll_mid") or 0.0)
    boll_lower = float(snapshot.get("boll_lower") or 0.0)
    boll_upper = float(snapshot.get("boll_upper") or 0.0)
    dcf_gap = snapshot.get("dcf_valuation_gap_pct")
    try:
        dcf_gap_pct = float(dcf_gap) if dcf_gap is not None else None
    except (TypeError, ValueError):
        dcf_gap_pct = None
    latest_volume_ratio = float(snapshot.get("latest_volume_ratio") or 1.0)
    market_environment = str(snapshot.get("market_environment") or "中性")
    market_bias_score = float(snapshot.get("market_bias_score") or 0.0)
    industry_environment = str(snapshot.get("industry_environment") or "中性")
    industry_bias_score = float(snapshot.get("industry_bias_score") or 0.0)
    earnings_phase = str(snapshot.get("earnings_phase") or "常规窗口")

    ranked_models = sorted(
        _load_quant_models(snapshot),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )[:4]
    top_scores = [float(item.get("score") or 0.0) for item in ranked_models]
    top_average = sum(top_scores) / len(top_scores) if top_scores else quant_probability
    bullish_models = sum(1 for score in top_scores if score >= 60)
    bearish_models = sum(1 for score in top_scores if score <= 40)
    dispersion = 0.0
    if top_scores:
        dispersion = math.sqrt(sum((score - top_average) ** 2 for score in top_scores) / len(top_scores))
    consensus_bonus = max(0.0, 1.6 - min(dispersion, 16.0) / 10.0)

    buy_level_raw = 1.0
    sell_level_raw = 1.0

    buy_level_raw += min(3.2, quant_probability / 30.0)
    sell_level_raw += min(3.2, max(0.0, 100.0 - quant_probability) / 30.0)

    buy_level_raw += min(1.8, top_average / 55.0)
    sell_level_raw += min(1.8, max(0.0, 100.0 - top_average) / 55.0)

    if ranked_models:
        buy_level_raw += min(1.0, bullish_models / len(ranked_models))
        sell_level_raw += min(1.0, bearish_models / len(ranked_models))
        buy_level_raw += consensus_bonus
        sell_level_raw += consensus_bonus

    buy_level_raw += min(2.0, float(action_summary.get("buy_score") or 0.0) / 4.0)
    sell_level_raw += min(2.0, float(action_summary.get("sell_score") or 0.0) / 4.0)

    if latest_price > 0 and ma_250 > 0:
        trend_gap_pct = (latest_price - ma_250) / ma_250 * 100
        if trend_gap_pct >= 0:
            buy_level_raw += min(0.9, 0.3 + trend_gap_pct / 12.0)
        else:
            sell_level_raw += min(0.9, 0.3 + abs(trend_gap_pct) / 12.0)

    if latest_price > 0 and boll_lower > 0 and latest_price <= boll_mid and latest_price >= boll_lower:
        buy_level_raw += 0.6
    if latest_price > 0 and boll_upper > 0 and latest_price >= boll_mid and latest_price >= boll_upper * 0.97:
        sell_level_raw += 0.6

    if dcf_gap_pct is not None:
        if dcf_gap_pct >= 0:
            buy_level_raw += min(1.0, dcf_gap_pct / 18.0)
            sell_level_raw -= min(0.6, dcf_gap_pct / 30.0)
        else:
            sell_level_raw += min(1.0, abs(dcf_gap_pct) / 18.0)
            buy_level_raw -= min(0.6, abs(dcf_gap_pct) / 30.0)

    if latest_volume_ratio >= 1.15:
        buy_level_raw += 0.5
    elif latest_volume_ratio < 0.8:
        buy_level_raw -= 0.6
        sell_level_raw += 0.4

    if market_environment == "偏强":
        buy_level_raw += min(0.8, 0.2 + max(0.0, market_bias_score) / 40.0)
    elif market_environment == "偏弱":
        buy_level_raw -= min(1.2, 0.4 + abs(market_bias_score) / 35.0)
        sell_level_raw += min(1.0, 0.3 + abs(market_bias_score) / 40.0)

    if industry_environment == "偏强":
        buy_level_raw += min(0.6, 0.15 + max(0.0, industry_bias_score) / 50.0)
    elif industry_environment == "偏弱":
        buy_level_raw -= min(0.8, 0.2 + abs(industry_bias_score) / 45.0)
        sell_level_raw += min(0.7, 0.15 + abs(industry_bias_score) / 55.0)

    if earnings_phase == "财报窗口进行中":
        buy_level_raw -= 0.8
        sell_level_raw += 0.5
    elif earnings_phase == "财报窗口临近":
        buy_level_raw -= 0.4
        sell_level_raw += 0.2

    action = str(action_summary.get("action") or "观望")
    if action == "偏买入":
        buy_level_raw += 0.8
        sell_level_raw -= 0.4
    elif action == "偏卖出":
        sell_level_raw += 0.8
        buy_level_raw -= 0.4

    buy_level = _clamp_level(buy_level_raw)
    sell_level = _clamp_level(sell_level_raw)
    return {
        "buy_recommendation_level": buy_level,
        "sell_recommendation_level": sell_level,
        "buy_recommendation_level_label": _level_label(buy_level),
        "sell_recommendation_level_label": _level_label(sell_level),
        "recommendation_level_method": (
            "参考多模型量化项目常见框架，综合四模型均分、一致性、显式买卖信号、趋势位置、成交量、大盘环境、行业强弱、财报节奏与 DCF 安全边际后得到 10 级评分。"
        ),
        "top_model_average_score": round(top_average, 2),
        "top_model_dispersion": round(dispersion, 2),
    }



def _positive_price_levels(*levels: float) -> list[float]:
    return [round(float(value), 2) for value in levels if float(value or 0.0) > 0]



def _deduplicate_prices(levels: list[float]) -> list[float]:
    unique: list[float] = []
    for value in sorted(levels):
        if not unique or abs(unique[-1] - value) > 0.01:
            unique.append(round(value, 2))
    return unique



def _format_price_range(levels: list[float], padding_pct: float = 0.012) -> str:
    cleaned = _deduplicate_prices(_positive_price_levels(*levels))
    if not cleaned:
        return "暂无明确价位"
    if len(cleaned) == 1:
        value = cleaned[0]
        return f"{value * (1 - padding_pct):.2f} - {value * (1 + padding_pct):.2f}"
    return f"{min(cleaned) * (1 - padding_pct):.2f} - {max(cleaned) * (1 + padding_pct):.2f}"



def _select_focus_levels(
    levels: list[float],
    reference_price: float,
    *,
    side: str,
    tolerance_pct: float = 0.035,
    max_pair_spread_pct: float = 0.02,
) -> list[float]:
    cleaned = _deduplicate_prices(levels)
    if not cleaned:
        return []
    if reference_price <= 0:
        return cleaned[:1]

    if side == "buy":
        candidates = [level for level in cleaned if level <= reference_price * (1 + tolerance_pct)] or cleaned[:]
        candidates.sort(key=lambda level: (abs(reference_price - level), -level))
    else:
        candidates = [level for level in cleaned if level >= reference_price * (1 - tolerance_pct)] or cleaned[:]
        candidates.sort(key=lambda level: (abs(reference_price - level), level))

    focus = [candidates[0]]
    for candidate in candidates[1:]:
        if len(focus) >= 2:
            break
        spread_pct = abs(candidate - focus[0]) / max(reference_price, focus[0], 1)
        if spread_pct <= max_pair_spread_pct:
            focus.append(candidate)
            break
    return sorted(_deduplicate_prices(focus))



def _format_focus_price_range(
    levels: list[float],
    reference_price: float,
    *,
    side: str,
    single_padding_pct: float = 0.006,
    pair_padding_pct: float = 0.004,
) -> str:
    focus_levels = _select_focus_levels(levels, reference_price, side=side)
    if not focus_levels:
        return "暂无明确价位"
    if len(focus_levels) == 1:
        value = focus_levels[0]
        return f"{value * (1 - single_padding_pct):.2f} - {value * (1 + single_padding_pct):.2f}"
    return f"{min(focus_levels) * (1 - pair_padding_pct):.2f} - {max(focus_levels) * (1 + pair_padding_pct):.2f}"



def build_recommended_price_plan(
    snapshot: Mapping[str, Any],
    signal_summary: Mapping[str, Any],
    position_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build explicit price plans for buy / sell / wait scenarios.

    这里不追求给一个“绝对准确”的单点价格，而是给出更适合实盘执行的区间：
    - 买入看支撑：250 日线、BOLL 中轨、BOLL 下轨
    - 卖出看压力：BOLL 上轨、成本保护、量化走弱后的反弹区
    - 观望看确认：向上确认位与向下防守位
    """

    latest_price = float(snapshot.get("latest_price") or 0.0)
    ma_250 = float(snapshot.get("ma_250") or 0.0)
    boll_mid = float(snapshot.get("boll_mid") or 0.0)
    boll_lower = float(snapshot.get("boll_lower") or 0.0)
    boll_upper = float(snapshot.get("boll_upper") or 0.0)
    average_cost = float(position_summary.get("average_cost") or 0.0)
    action = str(signal_summary.get("action") or "观望")

    support_levels = _deduplicate_prices(_positive_price_levels(boll_lower, ma_250, boll_mid))
    resistance_levels = _deduplicate_prices(
        _positive_price_levels(
            boll_upper,
            average_cost * 1.08 if average_cost > 0 else 0.0,
            latest_price * 1.05 if latest_price > 0 else 0.0,
        )
    )

    below_or_equal_supports = [level for level in support_levels if latest_price <= 0 or level <= latest_price]
    near_supports = below_or_equal_supports or [level for level in support_levels if level <= latest_price * 1.01] or support_levels[:1]
    primary_buy_levels = _select_focus_levels(near_supports, latest_price, side="buy")
    above_or_equal_resistances = [level for level in resistance_levels if level >= latest_price]
    near_resistances = above_or_equal_resistances or [level for level in resistance_levels if level >= latest_price * 0.995] or resistance_levels[-1:]
    primary_sell_levels = _select_focus_levels(near_resistances, latest_price, side="sell")

    watch_candidates = _positive_price_levels(boll_lower, ma_250, boll_mid, boll_upper)
    if action == "偏买入":
        buy_watch_candidates = near_supports or watch_candidates
        focus_watch_levels = _select_focus_levels(buy_watch_candidates, latest_price, side="buy", tolerance_pct=0.02, max_pair_spread_pct=0.012)
    elif action == "偏卖出":
        sell_watch_candidates = near_resistances or watch_candidates
        focus_watch_levels = _select_focus_levels(sell_watch_candidates, latest_price, side="sell", tolerance_pct=0.02, max_pair_spread_pct=0.012)
    else:
        nearest_watch = sorted(
            _deduplicate_prices(watch_candidates),
            key=lambda level: (abs(latest_price - level), abs(level - latest_price) / max(latest_price, 1)),
        )
        focus_watch_levels = _select_focus_levels(nearest_watch[:2], latest_price, side="buy" if latest_price <= (boll_mid or latest_price) else "sell", tolerance_pct=0.025, max_pair_spread_pct=0.012)

    breakout_watch = max(_positive_price_levels(ma_250, boll_mid, latest_price), default=0.0)
    defense_watch = min(_positive_price_levels(boll_lower, ma_250, boll_mid), default=0.0)

    buy_range = _format_focus_price_range(primary_buy_levels, latest_price, side="buy")
    sell_range = _format_focus_price_range(primary_sell_levels, latest_price, side="sell")
    watch_range = _format_focus_price_range(focus_watch_levels, latest_price, side="buy" if action != "偏卖出" else "sell", single_padding_pct=0.005, pair_padding_pct=0.003)
    suggested_add_price = round(max(primary_buy_levels), 2) if primary_buy_levels else 0.0
    suggested_reduce_price = round(min(primary_sell_levels), 2) if primary_sell_levels else 0.0
    stop_loss_base = min(_positive_price_levels(boll_lower, ma_250, average_cost, boll_mid), default=0.0)
    if stop_loss_base > 0:
        suggested_stop_loss_price = round(stop_loss_base * 0.97, 2)
    elif latest_price > 0:
        suggested_stop_loss_price = round(latest_price * 0.92, 2)
    else:
        suggested_stop_loss_price = 0.0

    buy_price_plan: list[str] = []
    sell_price_plan: list[str] = []
    watch_price_plan: list[str] = []

    if primary_buy_levels:
        buy_price_plan.append(f"推荐买入区间：{buy_range}，优先在支撑位附近分批挂单，不追高。")
        if boll_lower > 0:
            buy_price_plan.append(f"更激进的低吸价可关注 BOLL 下轨 {boll_lower:.2f} 附近。")
        if ma_250 > 0:
            buy_price_plan.append(f"若价格回踩并守住 250 日线 {ma_250:.2f}，可视为中线加仓确认。")
    else:
        buy_price_plan.append("当前缺少稳定支撑位，暂不建议主动抄底。")

    if primary_sell_levels:
        sell_price_plan.append(f"推荐卖出区间：{sell_range}，更适合分批止盈而不是一次性清仓。")
        if boll_upper > 0:
            sell_price_plan.append(f"若价格冲到 BOLL 上轨 {boll_upper:.2f} 附近且量能跟不上，可优先兑现一部分利润。")
        if average_cost > 0:
            sell_price_plan.append(f"若已有持仓，至少保证卖出价明显高于成本 {average_cost:.2f} 再做进攻型止盈。")
    else:
        sell_price_plan.append("当前上方明确压力位不足，卖出更应结合成本线与量化走弱信号。")

    if breakout_watch > 0:
        watch_price_plan.append(f"向上关注价：{breakout_watch:.2f}，站稳后再看是否形成新一轮上攻。")
    if defense_watch > 0:
        watch_price_plan.append(f"向下防守价：{defense_watch:.2f}，跌破后需重新评估仓位与节奏。")
    if action == "观望" and watch_range != "暂无明确价位":
        watch_price_plan.append(f"当前更适合观望，重点观察 {watch_range} 这一组高价值确认位。")
    elif action == "偏买入":
        watch_price_plan.append("当前偏买入，但仍应等待回踩确认，不建议直接追涨。")
    elif action == "偏卖出":
        watch_price_plan.append("当前偏卖出，若反弹未能重新站稳关键均线，优先执行减仓计划。")

    return {
        "recommended_buy_price_range": buy_range,
        "recommended_sell_price_range": sell_range,
        "watch_price_range": watch_range,
        "suggested_add_price": suggested_add_price,
        "suggested_reduce_price": suggested_reduce_price,
        "suggested_stop_loss_price": suggested_stop_loss_price,
        "buy_price_plan": buy_price_plan,
        "sell_price_plan": sell_price_plan,
        "watch_price_plan": watch_price_plan,
    }


def _neutral_snapshot_from_position(symbol: str, position_summary: Mapping[str, Any], snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(snapshot or {})
    payload.setdefault("symbol", symbol)
    payload.setdefault("display_name", symbol)
    payload.setdefault("latest_price", float(position_summary.get("average_cost") or 0.0))
    payload.setdefault("ma_250", 0.0)
    payload.setdefault("ma_30w", 0.0)
    payload.setdefault("ma_60w", 0.0)
    payload.setdefault("boll_mid", 0.0)
    payload.setdefault("boll_lower", 0.0)
    payload.setdefault("boll_upper", 0.0)
    payload.setdefault("dividend_yield", 0.0)
    payload.setdefault("quant_probability", 50.0)
    payload.setdefault("quant_model_breakdown", "[]")
    payload.setdefault("trigger_state", "正常")
    payload.setdefault("trigger_detail", "")
    payload.setdefault("updated_at", "")
    payload.setdefault("latest_volume_ratio", 1.0)
    payload.setdefault("market_environment", "中性")
    payload.setdefault("market_bias_score", 0.0)
    payload.setdefault("industry_name", "")
    payload.setdefault("industry_environment", "中性")
    payload.setdefault("industry_bias_score", 0.0)
    payload.setdefault("earnings_phase", "常规窗口")
    payload.setdefault("earnings_days_to_window", 999)
    return payload


def _extract_dcf_proxy_metrics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        breakdown = json.loads(str(snapshot.get("quant_model_breakdown") or "[]"))
    except json.JSONDecodeError:
        breakdown = []
    for item in breakdown:
        if str(item.get("key") or "") == "dcf_proxy":
            intrinsic_value = item.get("intrinsic_value")
            valuation_gap_pct = item.get("valuation_gap_pct")
            return {
                "dcf_intrinsic_value": round(float(intrinsic_value), 2) if intrinsic_value is not None else None,
                "dcf_valuation_gap_pct": round(float(valuation_gap_pct), 2) if valuation_gap_pct is not None else None,
                "dcf_label": str(item.get("label") or "DCF估值"),
                "dcf_reason": str(item.get("reason") or "DCF 估值数据暂不充分"),
            }
    return {
        "dcf_intrinsic_value": None,
        "dcf_valuation_gap_pct": None,
        "dcf_label": "DCF估值",
        "dcf_reason": "未启用 DCF 模型或当前估值数据不足",
    }


def _load_quant_models(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        breakdown = json.loads(str(snapshot.get("quant_model_breakdown") or "[]"))
    except json.JSONDecodeError:
        breakdown = []
    return [item for item in breakdown if isinstance(item, dict)]


def _build_quant_model_consensus(snapshot: Mapping[str, Any], limit: int = 4) -> tuple[str, str]:
    ranked_models = sorted(
        _load_quant_models(snapshot),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    top_models = ranked_models[:limit]
    if not top_models:
        return "多模型综合", "当前量化模型数据不足"

    average_score = sum(float(item.get("score") or 0.0) for item in top_models) / len(top_models)
    if average_score >= 80:
        consensus = "偏强"
    elif average_score >= 65:
        consensus = "中性偏强"
    elif average_score <= 45:
        consensus = "偏弱"
    else:
        consensus = "分歧较大"

    title = "四模型综合" if len(top_models) >= 4 else f"{len(top_models)}模型综合"
    details = "、".join(
        f"{str(item.get('label') or '模型')}{float(item.get('score') or 0.0):.0f}分"
        for item in top_models
    )
    return title, f"{title}{consensus}（{details}）"


def _format_signal_summary(signals: Sequence[str], prefix: str) -> str:
    cleaned = [str(item) for item in signals if str(item).strip()]
    if not cleaned:
        return f"{prefix}暂无显著信号"
    return f"{prefix}{'、'.join(cleaned)}"


def _build_watch_reason_lines(
    snapshot: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    price_plan: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    buy_score = float(action_summary.get("buy_score") or 0.0)
    sell_score = float(action_summary.get("sell_score") or 0.0)
    latest_volume_ratio = float(snapshot.get("latest_volume_ratio") or 1.0)
    market_environment = str(snapshot.get("market_environment") or "中性")
    industry_environment = str(snapshot.get("industry_environment") or "中性")
    earnings_phase = str(snapshot.get("earnings_phase") or "常规窗口")
    quant_probability = float(snapshot.get("quant_probability") or 0.0)
    watch_range = str(price_plan.get("watch_price_range") or "暂无明确价位")
    buy_signals = list(action_summary.get("buy_signals") or ())
    sell_signals = list(action_summary.get("sell_signals") or ())

    if abs(buy_score - sell_score) <= 2:
        if buy_signals and sell_signals:
            reasons.append("买入与卖出信号同时存在，多空仍在拉锯。")
        else:
            reasons.append("当前信号优势还不够明显，暂未形成单边结论。")
    if latest_volume_ratio < 0.85:
        reasons.append("成交量偏弱，缺少放量确认。")
    if market_environment == "偏弱" and industry_environment == "偏弱":
        reasons.append("大盘和行业都偏弱，外部环境仍在拖累。")
    elif market_environment == "偏弱":
        reasons.append("大盘环境偏弱，追价性价比不高。")
    elif industry_environment == "偏弱":
        reasons.append("所属行业暂未转强，先别急着放大仓位。")
    if earnings_phase != "常规窗口":
        reasons.append(f"{earnings_phase}，先防范事件波动。")
    if quant_probability < 60:
        reasons.append(f"量化综合盈利概率仅 {quant_probability:.2f}%，胜率优势不够突出。")
    if watch_range != "暂无明确价位":
        reasons.append(f"关键确认位集中在 {watch_range}。")
    return reasons[:4]


def _build_decision_summary(
    snapshot: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    price_plan: Mapping[str, Any],
) -> tuple[str, list[str]]:
    action = str(action_summary.get("action") or "观望")
    latest_price = float(snapshot.get("latest_price") or 0.0)
    buy_range = str(price_plan.get("recommended_buy_price_range") or "暂无明确价位")
    sell_range = str(price_plan.get("recommended_sell_price_range") or "暂无明确价位")
    watch_range = str(price_plan.get("watch_price_range") or "暂无明确价位")
    suggested_add_price = float(price_plan.get("suggested_add_price") or 0.0)
    suggested_reduce_price = float(price_plan.get("suggested_reduce_price") or 0.0)
    suggested_stop_loss_price = float(price_plan.get("suggested_stop_loss_price") or 0.0)

    if action == "偏买入":
        if suggested_add_price > 0 and latest_price <= suggested_add_price * 1.02:
            return (
                f"当前结论偏买入，现价已接近试仓区，可围绕 {buy_range} 分批布局。",
                [
                    "更适合分批试仓，不建议一次性重仓。",
                    f"若跌破 {suggested_stop_loss_price:.2f} 附近的防守位，需要重新评估节奏。" if suggested_stop_loss_price > 0 else "仍需保留止损纪律，避免左侧硬扛。",
                ],
            )
        if buy_range != "暂无明确价位":
            return (
                f"当前结论偏买入，但更稳妥的节奏是等回踩 {buy_range} 再分批介入。",
                ["当前不建议脱离支撑位追高，先等更好的风险收益比。"] if watch_range == "暂无明确价位" else [f"盘中重点看 {watch_range} 一带是否完成确认。"] ,
            )
        return ("当前结论偏买入，但支撑位还不够清晰，宜小仓位试探。", [])

    if action == "偏卖出":
        if suggested_reduce_price > 0 and latest_price >= suggested_reduce_price * 0.99:
            return (
                f"当前结论偏卖出，现价已接近减仓区，可按 {sell_range} 分批兑现。",
                [
                    "这代表优先处理仓位和风险，不代表必须一次性清仓。",
                    f"若后续直接跌破 {suggested_stop_loss_price:.2f} 附近的防守位，应先控风险。" if suggested_stop_loss_price > 0 else "若没有像样反弹，应优先控制回撤。",
                ],
            )
        if sell_range != "暂无明确价位":
            follow_line = (
                f"若先跌破 {suggested_stop_loss_price:.2f} 附近的防守位，则不必等反弹，先控风险。"
                if suggested_stop_loss_price > 0
                else "若后续没有反弹确认，应优先控制回撤。"
            )
            return (
                f"当前结论偏卖出，但现价还没到理想减仓区，若反弹到 {sell_range} 更适合分批减仓。",
                ["核心思路是借反弹优化卖点，而不是继续追买。", follow_line],
            )
        return ("当前结论偏卖出，优先降低弱势仓位和过重仓位。", [])

    watch_reasons = _build_watch_reason_lines(snapshot, action_summary, price_plan)
    if watch_range != "暂无明确价位":
        return (
            f"当前以观望为主，先等 {watch_range} 这一带给出方向。",
            watch_reasons,
        )
    return ("当前以观望为主，买卖信号尚未形成足够一致的优势。", watch_reasons)


def build_stock_comprehensive_advice(
    snapshot: Mapping[str, Any],
    position_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    symbol = str(snapshot.get("symbol") or "")
    resolved_position_summary = position_summary or {
        "average_cost": 0.0,
        "position_quantity": 0,
        "invalid_sell_quantity": 0,
    }
    resolved_snapshot = _neutral_snapshot_from_position(symbol, resolved_position_summary, snapshot)
    resolved_snapshot.update(_extract_dcf_proxy_metrics(resolved_snapshot))

    action_summary = build_market_action_summary(resolved_snapshot)
    price_plan = build_recommended_price_plan(resolved_snapshot, action_summary, resolved_position_summary)
    model_consensus_title, model_consensus_text = _build_quant_model_consensus(resolved_snapshot)
    recommendation_levels = _build_recommendation_levels(resolved_snapshot, action_summary)

    dcf_intrinsic_value = resolved_snapshot.get("dcf_intrinsic_value")
    dcf_valuation_gap_pct = resolved_snapshot.get("dcf_valuation_gap_pct")
    if dcf_intrinsic_value is not None:
        dcf_summary = (
            f"DCF 内在价值 {float(dcf_intrinsic_value):.2f}，"
            f"偏差 {float(dcf_valuation_gap_pct or 0.0):.2f}%"
        )
    else:
        dcf_summary = str(resolved_snapshot.get("dcf_reason") or "DCF 估值数据暂不充分")

    action = str(action_summary.get("action") or "观望")
    decision_summary, decision_reason_lines = _build_decision_summary(
        resolved_snapshot,
        action_summary,
        price_plan,
    )
    buy_signal_summary = _format_signal_summary(action_summary.get("buy_signals") or (), "买入侧信号：")
    sell_signal_summary = _format_signal_summary(action_summary.get("sell_signals") or (), "卖出侧信号：")
    market_context_line = (
        f"大盘{action_summary.get('market_environment', '中性')}"
        f" / 行业{action_summary.get('industry_environment', '中性')}"
        f" / 量能比 {float(action_summary.get('latest_volume_ratio') or 1.0):.2f}"
        f" / 财报节奏 {action_summary.get('earnings_phase', '常规窗口')}"
    )
    conclusion_line = f"{action}｜{model_consensus_text}；{market_context_line}；{decision_summary}"
    buy_line = (
        f"买点：{price_plan['recommended_buy_price_range']}｜"
        f"买入等级 {recommendation_levels['buy_recommendation_level']}/10"
        f"（{recommendation_levels['buy_recommendation_level_label']}）"
    )
    if price_plan["suggested_add_price"] > 0:
        buy_line += f"（参考加仓价 {price_plan['suggested_add_price']:.2f}）"
    if price_plan["watch_price_range"] != "暂无明确价位":
        buy_line += f"；关注位 {price_plan['watch_price_range']}"

    sell_line = (
        f"卖点：{price_plan['recommended_sell_price_range']}｜"
        f"卖出等级 {recommendation_levels['sell_recommendation_level']}/10"
        f"（{recommendation_levels['sell_recommendation_level_label']}）"
    )
    if price_plan["suggested_reduce_price"] > 0:
        sell_line += f"（参考减仓价 {price_plan['suggested_reduce_price']:.2f}）"
    if price_plan["suggested_stop_loss_price"] > 0:
        sell_line += f"；止损参考 {price_plan['suggested_stop_loss_price']:.2f}"

    dcf_line = f"DCF：{dcf_summary}"
    comprehensive_advice = "\n".join((
        f"结论：{conclusion_line}",
        f"原因：{action_summary['action_reason']}",
        buy_signal_summary,
        sell_signal_summary,
        *[f"观察：{item}" for item in decision_reason_lines],
        buy_line,
        sell_line,
        dcf_line,
    ))

    return {
        "action": action_summary["action"],
        "action_color": action_summary["action_color"],
        "action_reason": action_summary["action_reason"],
        "buy_signals": action_summary["buy_signals"],
        "sell_signals": action_summary["sell_signals"],
        "signal_groups": action_summary["signal_groups"],
        "buy_score": action_summary["buy_score"],
        "sell_score": action_summary["sell_score"],
        "dominant_model_label": action_summary["dominant_model_label"],
        "dominant_model_score": action_summary["dominant_model_score"],
        "recommended_buy_price_range": price_plan["recommended_buy_price_range"],
        "recommended_sell_price_range": price_plan["recommended_sell_price_range"],
        "watch_price_range": price_plan["watch_price_range"],
        "suggested_add_price": price_plan["suggested_add_price"],
        "suggested_reduce_price": price_plan["suggested_reduce_price"],
        "suggested_stop_loss_price": price_plan["suggested_stop_loss_price"],
        "buy_price_plan": price_plan["buy_price_plan"],
        "sell_price_plan": price_plan["sell_price_plan"],
        "watch_price_plan": price_plan["watch_price_plan"],
        "comprehensive_advice_title": model_consensus_title,
        "comprehensive_advice": comprehensive_advice,
        "advice_conclusion_line": conclusion_line,
        "advice_buy_line": buy_line,
        "advice_sell_line": sell_line,
        "advice_dcf_line": dcf_line,
        "decision_summary": decision_summary,
        "decision_reason_lines": decision_reason_lines,
        "buy_signal_summary": buy_signal_summary,
        "sell_signal_summary": sell_signal_summary,
        "model_consensus": model_consensus_text,
        "buy_recommendation_level": recommendation_levels["buy_recommendation_level"],
        "sell_recommendation_level": recommendation_levels["sell_recommendation_level"],
        "buy_recommendation_level_label": recommendation_levels["buy_recommendation_level_label"],
        "sell_recommendation_level_label": recommendation_levels["sell_recommendation_level_label"],
        "recommendation_level_method": recommendation_levels["recommendation_level_method"],
        "top_model_average_score": recommendation_levels["top_model_average_score"],
        "top_model_dispersion": recommendation_levels["top_model_dispersion"],
        "dcf_intrinsic_value": resolved_snapshot.get("dcf_intrinsic_value"),
        "dcf_valuation_gap_pct": resolved_snapshot.get("dcf_valuation_gap_pct"),
        "dcf_label": resolved_snapshot.get("dcf_label"),
        "dcf_reason": resolved_snapshot.get("dcf_reason"),
        "market_environment": action_summary.get("market_environment", "中性"),
        "market_bias_score": action_summary.get("market_bias_score", 0.0),
        "industry_environment": action_summary.get("industry_environment", "中性"),
        "industry_bias_score": action_summary.get("industry_bias_score", 0.0),
        "latest_volume_ratio": action_summary.get("latest_volume_ratio", 1.0),
        "earnings_phase": action_summary.get("earnings_phase", "常规窗口"),
    }


def _infer_position_style(snapshot: Mapping[str, Any], action_summary: Mapping[str, Any], weight_pct: float) -> str:
    quant_probability = float(snapshot.get("quant_probability") or 0.0)
    dividend_yield = float(snapshot.get("dividend_yield") or 0.0)
    ma_250 = float(snapshot.get("ma_250") or 0.0)
    latest_price = float(snapshot.get("latest_price") or 0.0)
    action = str(action_summary.get("action") or "观望")

    if weight_pct >= 40 and action == "偏买入":
        return "集中进攻型"
    if dividend_yield >= 4.5 and quant_probability >= 60:
        return "红利稳健型"
    if latest_price > 0 and ma_250 > 0 and latest_price >= ma_250 and quant_probability >= 75:
        return "趋势进攻型"
    if action == "偏卖出" or quant_probability <= 45:
        return "防守观察型"
    return "均衡配置型"


def _evaluate_symbol_risk(snapshot: Mapping[str, Any], action_summary: Mapping[str, Any], weight_pct: float) -> dict[str, Any]:
    latest_price = float(snapshot.get("latest_price") or 0.0)
    ma_250 = float(snapshot.get("ma_250") or 0.0)
    boll_mid = float(snapshot.get("boll_mid") or 0.0)
    quant_probability = float(snapshot.get("quant_probability") or 0.0)

    risk_score = 0
    reasons: list[str] = []
    if weight_pct >= 35:
        risk_score += 2
        reasons.append("单只仓位占比偏高")
    if ma_250 > 0 and latest_price < ma_250:
        risk_score += 1
        reasons.append("价格位于 250 日线下方")
    if boll_mid > 0 and latest_price < boll_mid:
        risk_score += 1
        reasons.append("价格位于 BOLL 中轨下方")
    if quant_probability <= 45:
        risk_score += 2
        reasons.append("量化评分偏弱")
    elif quant_probability <= 60:
        risk_score += 1
        reasons.append("量化评分一般")
    if str(action_summary.get("action")) == "偏卖出":
        risk_score += 2
        reasons.append("卖出信号占优")

    if risk_score >= 5:
        risk_level = "高"
    elif risk_score >= 3:
        risk_level = "中"
    else:
        risk_level = "低"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_reasons": reasons or ["当前未见明显高风险信号"],
    }


def _parse_ratio_range(ratio_range: str) -> tuple[float, float]:
    try:
        left, right = ratio_range.replace('%', '').split('-')
        return float(left.strip()), float(right.strip())
    except (TypeError, ValueError):
        return 0.0, 0.0


def _build_portfolio_adjustment_advice(
    holding_items: Sequence[Mapping[str, Any]],
    current_holding_ratio: float,
    target_ratio_range: str,
    overall_action: str,
) -> dict[str, Any]:
    low_ratio, high_ratio = _parse_ratio_range(target_ratio_range)
    target_mid = round((low_ratio + high_ratio) / 2, 2) if high_ratio > 0 else current_holding_ratio

    overall_suggestions: list[str] = []
    priority_reduce: list[str] = []
    priority_add: list[str] = []

    if current_holding_ratio > high_ratio > 0:
        overall_suggestions.append(
            f"当前总仓位约 {current_holding_ratio:.2f}% ，高于建议区间 {target_ratio_range}，更适合逐步降到约 {target_mid:.2f}% 附近。"
        )
    elif 0 < current_holding_ratio < low_ratio:
        overall_suggestions.append(
            f"当前总仓位约 {current_holding_ratio:.2f}% ，低于建议区间 {target_ratio_range}，若强势信号延续，可逐步提升到约 {target_mid:.2f}% 附近。"
        )
    else:
        overall_suggestions.append(
            f"当前总仓位约 {current_holding_ratio:.2f}% ，与建议区间 {target_ratio_range} 基本匹配，可先维持节奏。"
        )

    reduce_candidates = sorted(
        [item for item in holding_items if item.get("action") == "偏卖出" or item.get("risk_level") == "高"],
        key=lambda item: (-float(item.get("weight_pct") or 0.0), -float(item.get("risk_score") or 0.0), str(item.get("symbol") or "")),
    )
    add_candidates = sorted(
        [
            item for item in holding_items
            if item.get("action") == "偏买入" and item.get("risk_level") != "高"
        ],
        key=lambda item: (-float(item.get("quant_probability") or 0.0), float(item.get("weight_pct") or 0.0), str(item.get("symbol") or "")),
    )

    for item in reduce_candidates[:3]:
        priority_reduce.append(
            f"优先考虑减仓 {item.get('display_name') or item['symbol']}（占比 {item['weight_pct']:.2f}%），参考减仓价 {item['suggested_reduce_price']:.2f}，原因：{item['action_reason']}"
        )
    for item in add_candidates[:3]:
        priority_add.append(
            f"若市场继续转强，可优先关注加仓 {item.get('display_name') or item['symbol']}（当前占比 {item['weight_pct']:.2f}%），参考加仓价 {item['suggested_add_price']:.2f}。"
        )

    if overall_action == "偏卖出" and not priority_reduce:
        overall_suggestions.append("当前组合虽偏防守，但暂无特别突出的单一减仓对象，可先从高权重仓位小幅收缩。")
    if overall_action == "偏买入" and not priority_add:
        overall_suggestions.append("当前组合偏强，但没有特别明确的低风险加仓对象，建议继续等待回踩确认。")

    return {
        "target_holding_ratio_mid": target_mid,
        "overall_suggestions": overall_suggestions,
        "priority_reduce": priority_reduce,
        "priority_add": priority_add,
    }


def _build_portfolio_professional_advice(
    holding_items: Sequence[Mapping[str, Any]],
    overall_action: str,
    risk_level: str,
    weighted_quant_probability: float,
    recommended_holding_ratio: str,
) -> list[str]:
    strongest = [item for item in holding_items if item.get("action") == "偏买入"]
    weakest = [item for item in holding_items if item.get("action") == "偏卖出" or item.get("risk_level") == "高"]
    strongest_names = "、".join(str(item.get("display_name") or item.get("symbol") or "-") for item in strongest[:3]) or "暂无明显强势仓位"
    weakest_names = "、".join(str(item.get("display_name") or item.get("symbol") or "-") for item in weakest[:3]) or "暂无明显弱势仓位"
    if weighted_quant_probability >= 75:
        quant_view = "组合量化评分整体偏强"
    elif weighted_quant_probability <= 45:
        quant_view = "组合量化评分整体偏弱"
    else:
        quant_view = "组合量化评分处于中性区间"
    return [
        f"专业观点：当前组合判断为{overall_action}，风险等级为{risk_level}，建议总仓位参考 {recommended_holding_ratio}。",
        f"量化视角：{quant_view}，可优先跟踪强势持仓 {strongest_names}。",
        f"调仓重点：当前更需要关注 {weakest_names} 的仓位和节奏，避免弱势仓位拖累整体表现。",
    ]


def _recommended_holding_ratio_range(overall_action: str, risk_level: str, weighted_quant_probability: float) -> str:
    if overall_action == "偏卖出" or risk_level == "高":
        return "20% - 40%"
    if overall_action == "观望":
        return "40% - 60%"
    if weighted_quant_probability >= 80 and risk_level == "低":
        return "65% - 85%"
    return "50% - 70%"


def build_portfolio_profile(
    trade_records: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    total_investment_amount: float | None = None,
) -> dict[str, Any]:
    """Build an account-level holdings profile from recorded trades and latest snapshots.

    这部分回答的是“我现在整体仓位怎么样”：
    - 当前到底持有哪些股票、每只占多少
    - 风格更偏进攻、均衡还是防守
    - 组合整体风险高不高
    - 模型视角下，更合适的仓位区间是多少
    """

    grouped_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in trade_records:
        grouped_records[str(trade["symbol"])] .append(trade)

    snapshot_map = {str(item["symbol"]): dict(item) for item in snapshots}
    holding_items: list[dict[str, Any]] = []
    closed_items: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_realized_pnl = 0.0
    total_buy_amount = 0.0
    total_sell_amount = 0.0

    for symbol, rows in grouped_records.items():
        position_summary = build_position_summary(rows)
        total_realized_pnl += float(position_summary["realized_pnl"])
        total_buy_amount += float(position_summary["total_buy_amount"])
        total_sell_amount += float(position_summary["total_sell_amount"])
        position_quantity = int(position_summary["position_quantity"])
        if position_quantity <= 0:
            if float(position_summary["total_sell_amount"]) > 0:
                snapshot = _neutral_snapshot_from_position(symbol, position_summary, snapshot_map.get(symbol))
                closed_items.append(
                    {
                        "symbol": symbol,
                        "display_name": str(snapshot.get("display_name") or symbol),
                        "realized_pnl": float(position_summary["realized_pnl"]),
                        "total_buy_amount": float(position_summary["total_buy_amount"]),
                        "total_sell_amount": float(position_summary["total_sell_amount"]),
                        "total_commission_fee": float(position_summary.get("total_commission_fee") or 0.0),
                        "trade_count": int(position_summary["trade_count"]),
                        "last_traded_at": max((str(item.get("traded_at") or "") for item in rows), default=""),
                    }
                )
            continue

        snapshot = _neutral_snapshot_from_position(symbol, position_summary, snapshot_map.get(symbol))
        snapshot.update(_extract_dcf_proxy_metrics(snapshot))
        latest_price = float(snapshot.get("latest_price") or position_summary["average_cost"] or 0.0)
        cost_basis = float(position_summary["cost_basis_total"])
        market_value = round(latest_price * position_quantity, 2)
        unrealized_pnl = round(market_value - cost_basis, 2)
        unrealized_pnl_pct = round((unrealized_pnl / cost_basis) * 100, 2) if cost_basis > 0 else 0.0
        total_market_value += market_value
        total_cost_basis += cost_basis

        holding_items.append(
            {
                "symbol": symbol,
                "display_name": str(snapshot.get("display_name") or symbol),
                "position_quantity": position_quantity,
                "average_cost": float(position_summary["average_cost"]),
                "cost_basis_total": cost_basis,
                "latest_price": latest_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "quant_probability": float(snapshot.get("quant_probability") or 0.0),
                "dividend_yield": float(snapshot.get("dividend_yield") or 0.0),
                "snapshot": snapshot,
                "position_summary": position_summary,
            }
        )

    if not holding_items:
        return {
            "has_positions": False,
            "holding_ratio": 0.0,
            "total_investment_amount": round(float(total_investment_amount or 0.0), 2),
            "recommended_holding_ratio": "0% - 20%",
            "holding_style": "空仓观察型",
            "risk_level": "低",
            "risk_score": 0,
            "overall_action": "观望",
            "target_holding_ratio_mid": 10.0,
            "weighted_quant_probability": 0.0,
            "weighted_dividend_yield": 0.0,
            "total_market_value": 0.0,
            "total_cost_basis": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_unrealized_pnl_pct": 0.0,
            "total_realized_pnl": round(total_realized_pnl, 2),
            "active_positions": [],
            "closed_positions": closed_items,
            "comprehensive_advice": "当前没有持仓，建议先等待更明确的买入信号，再逐步建立试探仓位。",
            "overall_adjustment_suggestions": ["当前空仓，建议先从小仓位试探，不必急于满仓。"],
            "priority_reduce_positions": [],
            "priority_add_positions": [],
            "professional_advice": ["当前账户没有持仓，可先设置股市总投入金额并等待更明确的买入信号。"],
            "risk_reasons": ["当前账户没有在途仓位，整体回撤压力较低。"],
            "analysis_note": "若已填写股市总投入金额，持仓比例会按当前持仓市值 / 总投入金额计算。",
        }

    for item in holding_items:
        weight_pct = round(item["market_value"] / total_market_value * 100, 2) if total_market_value > 0 else 0.0
        snapshot = item["snapshot"]
        display_advice = build_stock_comprehensive_advice(snapshot, item["position_summary"])
        action_summary = {
            "action": display_advice["action"],
            "action_color": display_advice["action_color"],
            "action_reason": display_advice["action_reason"],
        }
        risk_info = _evaluate_symbol_risk(snapshot, action_summary, weight_pct)
        item["weight_pct"] = weight_pct
        item["action"] = action_summary["action"]
        item["action_color"] = action_summary["action_color"]
        item["action_reason"] = action_summary["action_reason"]
        item["position_style"] = _infer_position_style(snapshot, action_summary, weight_pct)
        item["dcf_intrinsic_value"] = display_advice["dcf_intrinsic_value"]
        item["dcf_valuation_gap_pct"] = display_advice["dcf_valuation_gap_pct"]
        item["dcf_label"] = display_advice["dcf_label"]
        item["dcf_reason"] = display_advice["dcf_reason"]
        item["recommended_buy_price_range"] = display_advice["recommended_buy_price_range"]
        item["recommended_sell_price_range"] = display_advice["recommended_sell_price_range"]
        item["add_price_range"] = display_advice["recommended_buy_price_range"]
        item["reduce_price_range"] = display_advice["recommended_sell_price_range"]
        item["watch_price_range"] = display_advice["watch_price_range"]
        item["suggested_add_price"] = display_advice["suggested_add_price"]
        item["suggested_reduce_price"] = display_advice["suggested_reduce_price"]
        item["suggested_stop_loss_price"] = display_advice["suggested_stop_loss_price"]
        item["buy_recommendation_level"] = display_advice["buy_recommendation_level"]
        item["sell_recommendation_level"] = display_advice["sell_recommendation_level"]
        item["buy_recommendation_level_label"] = display_advice["buy_recommendation_level_label"]
        item["sell_recommendation_level_label"] = display_advice["sell_recommendation_level_label"]
        item["comprehensive_advice"] = display_advice["comprehensive_advice"]
        item["comprehensive_advice_title"] = display_advice["comprehensive_advice_title"]
        item["advice_conclusion_line"] = display_advice["advice_conclusion_line"]
        item["advice_buy_line"] = display_advice["advice_buy_line"]
        item["advice_sell_line"] = display_advice["advice_sell_line"]
        item["advice_dcf_line"] = display_advice["advice_dcf_line"]
        item["model_consensus"] = display_advice["model_consensus"]
        item["decision_summary"] = display_advice["decision_summary"]
        item["decision_reason_lines"] = display_advice["decision_reason_lines"]
        item["buy_signal_summary"] = display_advice["buy_signal_summary"]
        item["sell_signal_summary"] = display_advice["sell_signal_summary"]
        item["buy_price_plan"] = list(display_advice["buy_price_plan"])
        item["sell_price_plan"] = list(display_advice["sell_price_plan"])
        item["watch_price_plan"] = list(display_advice["watch_price_plan"])
        specific_advice_lines: list[str] = []
        for advice_line in (
            display_advice["decision_summary"],
            *display_advice["decision_reason_lines"],
            *(display_advice["buy_price_plan"][:1]),
            *(display_advice["sell_price_plan"][:1]),
            *(display_advice["watch_price_plan"][:1]),
        ):
            cleaned_line = str(advice_line or "").strip()
            if cleaned_line and cleaned_line not in specific_advice_lines:
                specific_advice_lines.append(cleaned_line)
        item["specific_advice_lines"] = specific_advice_lines[:5]
        item.update(risk_info)

    holding_items.sort(key=lambda entry: (-entry["weight_pct"], entry["symbol"]))
    closed_items.sort(
        key=lambda entry: (
            str(entry.get("last_traded_at") or ""),
            str(entry.get("symbol") or ""),
        ),
        reverse=True,
    )

    weighted_quant_probability = round(
        sum(item["quant_probability"] * item["weight_pct"] for item in holding_items) / 100,
        2,
    )
    weighted_dividend_yield = round(
        sum(item["dividend_yield"] * item["weight_pct"] for item in holding_items) / 100,
        2,
    )
    total_unrealized_pnl = round(total_market_value - total_cost_basis, 2)
    total_unrealized_pnl_pct = round((total_unrealized_pnl / total_cost_basis) * 100, 2) if total_cost_basis > 0 else 0.0
    normalized_total_investment_amount = round(max(float(total_investment_amount or 0.0), 0.0), 2)
    estimated_portfolio_base = total_market_value + max(total_sell_amount, 0.0)
    base_amount = normalized_total_investment_amount if normalized_total_investment_amount > 0 else estimated_portfolio_base
    holding_ratio = round((total_market_value / base_amount) * 100, 2) if base_amount > 0 else 0.0

    portfolio_buy_score = round(sum(item["market_value"] * build_market_action_summary(item["snapshot"])["buy_score"] for item in holding_items) / max(total_market_value, 1), 2)
    portfolio_sell_score = round(sum(item["market_value"] * build_market_action_summary(item["snapshot"])["sell_score"] for item in holding_items) / max(total_market_value, 1), 2)

    if portfolio_buy_score - portfolio_sell_score >= 2.5:
        overall_action = "偏买入"
    elif portfolio_sell_score - portfolio_buy_score >= 2.5:
        overall_action = "偏卖出"
    else:
        overall_action = "观望"

    top_weight = max(item["weight_pct"] for item in holding_items)
    weak_position_weight = sum(item["weight_pct"] for item in holding_items if item["risk_level"] == "高" or item["action"] == "偏卖出")
    portfolio_risk_score = 0
    portfolio_risk_reasons: list[str] = []
    if holding_ratio >= 80:
        portfolio_risk_score += 2
        portfolio_risk_reasons.append("整体持仓比例偏高")
    elif holding_ratio >= 65:
        portfolio_risk_score += 1
        portfolio_risk_reasons.append("整体仓位不低，需要控制回撤")
    if top_weight >= 45:
        portfolio_risk_score += 2
        portfolio_risk_reasons.append("单一股票仓位集中度偏高")
    elif top_weight >= 30:
        portfolio_risk_score += 1
        portfolio_risk_reasons.append("前排重仓股占比不低")
    if weak_position_weight >= 35:
        portfolio_risk_score += 2
        portfolio_risk_reasons.append("弱势或偏卖出仓位占比偏高")
    elif weak_position_weight >= 20:
        portfolio_risk_score += 1
        portfolio_risk_reasons.append("组合中已有一定比例弱势仓位")
    if weighted_quant_probability <= 50:
        portfolio_risk_score += 2
        portfolio_risk_reasons.append("组合加权量化评分偏弱")
    elif weighted_quant_probability <= 65:
        portfolio_risk_score += 1
        portfolio_risk_reasons.append("组合加权量化评分一般")

    if portfolio_risk_score >= 5:
        risk_level = "高"
    elif portfolio_risk_score >= 3:
        risk_level = "中"
    else:
        risk_level = "低"

    if weighted_dividend_yield >= 4.5 and holding_ratio <= 70:
        holding_style = "红利稳健型"
    elif top_weight >= 45 and holding_ratio >= 70:
        holding_style = "集中进攻型"
    elif overall_action == "偏买入" and weighted_quant_probability >= 75:
        holding_style = "趋势进攻型"
    elif holding_ratio <= 35:
        holding_style = "防守观望型"
    else:
        holding_style = "均衡配置型"

    recommended_holding_ratio = _recommended_holding_ratio_range(overall_action, risk_level, weighted_quant_probability)

    if overall_action == "偏卖出":
        comprehensive_advice = (
            f"当前组合更偏防守，建议把总仓位逐步收缩到 {recommended_holding_ratio}，"
            "优先处理弱势仓位和高集中度仓位。"
        )
    elif overall_action == "偏买入" and risk_level == "低":
        comprehensive_advice = (
            f"当前组合相对健康，可围绕强势仓位把总仓位维持在 {recommended_holding_ratio}，"
            "但仍建议分批而不是一次性加满。"
        )
    else:
        comprehensive_advice = (
            f"当前组合多空信号并存，建议先把总仓位控制在 {recommended_holding_ratio}，"
            "等待强弱分化更明确后再调整。"
        )

    adjustment_advice = _build_portfolio_adjustment_advice(
        holding_items,
        holding_ratio,
        recommended_holding_ratio,
        overall_action,
    )
    professional_advice = _build_portfolio_professional_advice(
        holding_items,
        overall_action,
        risk_level,
        weighted_quant_probability,
        recommended_holding_ratio,
    )

    return {
        "has_positions": True,
        "holding_ratio": holding_ratio,
        "total_investment_amount": normalized_total_investment_amount,
        "recommended_holding_ratio": recommended_holding_ratio,
        "holding_style": holding_style,
        "risk_level": risk_level,
        "risk_score": portfolio_risk_score,
        "overall_action": overall_action,
        "weighted_quant_probability": weighted_quant_probability,
        "weighted_dividend_yield": weighted_dividend_yield,
        "total_market_value": round(total_market_value, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
        "total_realized_pnl": round(total_realized_pnl, 2),
        "active_positions": holding_items,
        "closed_positions": closed_items,
        "comprehensive_advice": comprehensive_advice,
        "target_holding_ratio_mid": adjustment_advice["target_holding_ratio_mid"],
        "overall_adjustment_suggestions": adjustment_advice["overall_suggestions"],
        "priority_reduce_positions": adjustment_advice["priority_reduce"],
        "priority_add_positions": adjustment_advice["priority_add"],
        "professional_advice": professional_advice,
        "risk_reasons": portfolio_risk_reasons or ["组合结构暂时平衡，没有明显超额风险信号。"],
        "analysis_note": (
            "持仓比例按当前持仓市值 / 股市总投入金额计算。" if normalized_total_investment_amount > 0
            else "尚未填写股市总投入金额，当前持仓比例先按已记录交易资金估算：当前持仓市值 /（当前持仓市值 + 历史卖出回笼资金）。"
        ),
    }



@dataclass(frozen=True)
class TradeAnalysisResult:
    """Normalized analysis result regardless of provider success or fallback."""

    provider: str
    model_name: str
    status: str
    analysis: dict[str, Any]
    error_message: str | None = None


def build_position_summary(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild the current position from a buy/sell journal."""

    position_quantity = 0
    cost_basis_total = 0.0
    realized_pnl = 0.0
    total_buy_amount = 0.0
    total_sell_amount = 0.0
    total_commission_fee = 0.0
    invalid_sell_quantity = 0

    for trade in trades:
        side = str(trade["side"])
        quantity = int(trade["quantity"])
        price = float(trade["price"])
        amount = price * quantity
        commission_fee = float(trade.get("commission_fee") or max(amount * 0.0003, 5.0))
        total_commission_fee += commission_fee

        if side == "buy":
            total_buy_amount += amount + commission_fee
            cost_basis_total += amount + commission_fee
            position_quantity += quantity
            continue

        total_sell_amount += amount - commission_fee
        if position_quantity <= 0:
            invalid_sell_quantity += quantity
            continue

        matched_quantity = min(quantity, position_quantity)
        average_cost = cost_basis_total / position_quantity if position_quantity else 0.0
        matched_fee = commission_fee * (matched_quantity / quantity) if quantity > 0 else 0.0
        realized_pnl += (price - average_cost) * matched_quantity - matched_fee
        cost_basis_total -= average_cost * matched_quantity
        position_quantity -= matched_quantity
        if quantity > matched_quantity:
            invalid_sell_quantity += quantity - matched_quantity

    average_cost = cost_basis_total / position_quantity if position_quantity else 0.0
    return {
        "position_quantity": position_quantity,
        "average_cost": round(average_cost, 2),
        "cost_basis_total": round(cost_basis_total, 2),
        "realized_pnl": round(realized_pnl, 2),
        "total_buy_amount": round(total_buy_amount, 2),
        "total_sell_amount": round(total_sell_amount, 2),
        "total_commission_fee": round(total_commission_fee, 2),
        "trade_count": len(trades),
        "invalid_sell_quantity": invalid_sell_quantity,
    }


class TradeAdvisor:
    """Generate trade rationality analysis and next-step suggestions.

    这里故意把“分析”从路由层拆开，避免 `app.py` 里混入太多提示词和规则细节。
    """
    """Generate trade rationality analysis and next-step suggestions."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def analyze_symbol(
        self,
        symbol: str,
        snapshot: Mapping[str, Any],
        trades: Sequence[Mapping[str, Any]],
    ) -> TradeAnalysisResult:
        """Try the configured LLM first and fall back to a rule-based summary."""

        position_summary = build_position_summary(trades)
        market_snapshot = {
            "symbol": symbol,
            "display_name": snapshot.get("display_name") or symbol,
            "latest_price": float(snapshot.get("latest_price") or 0.0),
            "ma_250": float(snapshot.get("ma_250") or 0.0),
            "ma_30w": float(snapshot.get("ma_30w") or 0.0),
            "ma_60w": float(snapshot.get("ma_60w") or 0.0),
            "boll_mid": float(snapshot.get("boll_mid") or 0.0),
            "boll_lower": float(snapshot.get("boll_lower") or 0.0),
            "boll_upper": float(snapshot.get("boll_upper") or 0.0),
            "dividend_yield": float(snapshot.get("dividend_yield") or 0.0),
            "quant_probability": float(snapshot.get("quant_probability") or 0.0),
            "quant_model_breakdown": snapshot.get("quant_model_breakdown") or "[]",
            "trigger_state": snapshot.get("trigger_state") or "正常",
            "trigger_detail": snapshot.get("trigger_detail") or "",
            "updated_at": snapshot.get("updated_at") or "",
        }

        if self.settings.llm_provider_name == "openai" and self.settings.llm_api_key:
            try:
                return self._analyze_with_openai(
                    symbol=symbol,
                    market_snapshot=market_snapshot,
                    position_summary=position_summary,
                    trades=trades,
                )
            except Exception as exc:  # pragma: no cover - network side effect
                fallback = self._rule_based_analysis(market_snapshot, position_summary)
                fallback.update(build_stock_comprehensive_advice(market_snapshot, position_summary))
                return TradeAnalysisResult(
                    provider="rule-based",
                    model_name="local-fallback",
                    status="fallback",
                    analysis=fallback,
                    error_message=str(exc),
                )

        fallback_analysis = self._rule_based_analysis(market_snapshot, position_summary)
        fallback_analysis.update(build_stock_comprehensive_advice(market_snapshot, position_summary))
        return TradeAnalysisResult(
            provider="rule-based",
            model_name="local-fallback",
            status="fallback",
            analysis=fallback_analysis,
            error_message="未配置 OPENAI_API_KEY，已回退到规则分析",
        )

    def _analyze_with_openai(
        self,
        symbol: str,
        market_snapshot: dict[str, Any],
        position_summary: dict[str, Any],
        trades: Sequence[Mapping[str, Any]],
    ) -> TradeAnalysisResult:
        """Call the OpenAI Responses API and request a structured JSON answer."""

        # 这里组装的是给大模型看的完整上下文：
        # 持仓摘要 + 当前快照 + 逐笔交易流水。
        prompt = {
            "symbol": symbol,
            "market_snapshot": market_snapshot,
            "position_summary": position_summary,
            "trades": [
                {
                    "side": str(item["side"]),
                    "price": float(item["price"]),
                    "quantity": int(item["quantity"]),
                    "traded_at": str(item["traded_at"]),
                    "note": str(item["note"] if "note" in item.keys() else ""),
                }
                for item in trades
            ],
        }
        schema = {
            "name": "trade_analysis",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "judgment": {
                        "type": "string",
                        "enum": ["合理", "偏激进", "偏保守", "不合理", "偏买入", "偏卖出", "观望"],
                    },
                    "reasoning": {"type": "array", "items": {"type": "string"}},
                    "next_buy_points": {"type": "array", "items": {"type": "string"}},
                    "next_sell_points": {"type": "array", "items": {"type": "string"}},
                    "watch_points": {"type": "array", "items": {"type": "string"}},
                    "recommended_buy_price_range": {"type": "string"},
                    "recommended_sell_price_range": {"type": "string"},
                    "watch_price_range": {"type": "string"},
                    "risk_controls": {"type": "array", "items": {"type": "string"}},
                    "position_advice": {"type": "string"},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    "disclaimer": {"type": "string"},
                },
                "required": [
                    "summary",
                    "judgment",
                    "reasoning",
                    "next_buy_points",
                    "next_sell_points",
                    "watch_points",
                    "recommended_buy_price_range",
                    "recommended_sell_price_range",
                    "watch_price_range",
                    "risk_controls",
                    "position_advice",
                    "confidence",
                    "disclaimer",
                ],
            },
            "strict": True,
        }
        # 当前接的是 OpenAI Responses API，并要求模型严格输出 JSON。
        response = requests.post(
            self.settings.llm_base_url,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.llm_model_name,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "你是谨慎的股票交易复盘助手。"
                                    "仅基于给定的持仓和行情信息做风险分析，"
                                    "不要承诺收益，不要给出确定性投资结论。"
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "请分析以下 A 股交易记录是否合理，并给出下一步更稳健的买卖点建议，以及明确的推荐买入价格、推荐卖出价格、观望关注价格。"
                                    "输出必须符合 JSON Schema。\n"
                                    f"{json.dumps(prompt, ensure_ascii=False)}"
                                ),
                            }
                        ],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema["name"],
                        "schema": schema["schema"],
                        "strict": schema["strict"],
                    }
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = self._extract_output_text(payload)
        analysis = json.loads(raw_text)
        analysis.update(build_stock_comprehensive_advice(market_snapshot, position_summary))
        return TradeAnalysisResult(
            provider="openai",
            model_name=self.settings.llm_model_name,
            status="success",
            analysis=analysis,
        )

    @staticmethod
    def _extract_output_text(payload: Mapping[str, Any]) -> str:
        """Support both top-level and nested response text formats."""

        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        fragments: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    fragments.append(str(content["text"]))
        if not fragments:
            raise ValueError("LLM response does not contain output_text")
        return "\n".join(fragments)

    @staticmethod
    def _rule_based_analysis(
        market_snapshot: Mapping[str, Any],
        position_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Provide a deterministic fallback when LLM access is unavailable."""

        latest_price = float(market_snapshot["latest_price"])
        average_cost = float(position_summary["average_cost"])
        dividend_yield = float(market_snapshot["dividend_yield"])
        ma_250 = float(market_snapshot["ma_250"])
        boll_mid = float(market_snapshot["boll_mid"])
        quant_probability = float(market_snapshot.get("quant_probability") or 0.0)
        position_quantity = int(position_summary["position_quantity"])
        signal_summary = build_market_action_summary(market_snapshot)
        price_plan = build_recommended_price_plan(market_snapshot, signal_summary, position_summary)

        reasoning: list[str] = []
        next_buy_points: list[str] = []
        next_sell_points: list[str] = []
        watch_points: list[str] = list(price_plan["watch_price_plan"])
        risk_controls: list[str] = []

        # 规则分析的目标不是代替投顾，而是在没有 LLM 时给出基础复盘建议。
        if position_quantity == 0:
            reasoning.append("当前没有持仓，更适合把重点放在分批建仓和风险预算上。")
        else:
            reasoning.append(
                f"当前持仓 {position_quantity} 股，持仓均价约 {average_cost:.2f}。"
            )

        reasoning.append(signal_summary["action_reason"])
        reasoning.append(
            f"推荐买入价区间 {price_plan['recommended_buy_price_range']}，"
            f"推荐卖出价区间 {price_plan['recommended_sell_price_range']}。"
        )

        if ma_250 and latest_price < ma_250:
            reasoning.append("现价低于 250 日线，趋势偏弱，追高并不划算。")
        else:
            reasoning.append("现价位于 250 日线之上，中长期趋势相对更稳。")

        if dividend_yield >= 4.5:
            reasoning.append("股息率较高，若基本面稳定，可作为观察加分项。")
        elif dividend_yield < 3.5:
            reasoning.append("股息率已经偏低，防守型持有价值下降。")

        if quant_probability >= 85:
            reasoning.append("量化综合评分处于高位，说明多因子共振更偏向买入侧。")
        elif quant_probability <= 40:
            reasoning.append("量化综合评分偏低，说明当前更需要先考虑风控而不是继续进攻。")

        next_buy_points.extend(price_plan["buy_price_plan"])
        next_sell_points.extend(price_plan["sell_price_plan"])

        if boll_mid and latest_price < boll_mid:
            next_sell_points.append(f"若反弹至 BOLL 中轨 {boll_mid:.2f} 附近但量能不足，可考虑先减一部分仓位。")
            risk_controls.append("弱势区间内不要一次性满仓抄底，优先分批建仓。")
        else:
            next_sell_points.append("若后续冲高但无法放量突破前高，可沿上方压力带分批止盈。")

        if position_quantity > 0 and average_cost > 0:
            if latest_price < average_cost:
                reasoning.append("当前处于浮亏状态，更适合先管控仓位，而不是情绪化补仓。")
                risk_controls.append("单次补仓不建议超过现有仓位的三分之一。")
                judgment = "偏激进"
            else:
                reasoning.append("当前浮盈或接近盈亏平衡，操作弹性更大。")
                judgment = "合理"
        else:
            judgment = "偏保守"

        if signal_summary["action"] == "偏卖出":
            judgment = "偏卖出"
        elif signal_summary["action"] == "偏买入" and judgment != "不合理":
            judgment = "偏买入"
        elif signal_summary["action"] == "观望" and judgment not in {"不合理", "偏激进"}:
            judgment = "观望"

        if int(position_summary["invalid_sell_quantity"]) > 0:
            reasoning.append("存在超出持仓的卖出记录，请先核对交易流水。")
            judgment = "不合理"
            risk_controls.append("卖出数量应小于等于当前实际持仓。")

        if signal_summary["action"] == "偏卖出":
            position_advice = (
                f"当前卖出信号占优，优先考虑在 {price_plan['recommended_sell_price_range']} 区间分批止盈或减仓，"
                "避免把已有利润重新回吐。"
            )
        elif signal_summary["action"] == "偏买入":
            position_advice = (
                f"当前买入信号占优，可重点盯住 {price_plan['recommended_buy_price_range']} 区间分批低吸，"
                "但不建议一次性重仓追入。"
            )
        else:
            position_advice = (
                f"当前买卖信号并存，建议先观望，重点观察 {price_plan['watch_price_range']} 这组关键价位，"
                "等待趋势或量化评分进一步拉开差距。"
            )

        if not risk_controls:
            risk_controls.append("控制单笔风险，优先使用预设止损和分批执行。")

        return {
            "summary": (
                f"已结合买入/卖出规则与量化多因子完成综合判断，当前建议：{signal_summary['action']}；"
                f"推荐买入价 {price_plan['recommended_buy_price_range']}，"
                f"推荐卖出价 {price_plan['recommended_sell_price_range']}。"
            ),
            "judgment": judgment,
            "reasoning": reasoning,
            "next_buy_points": next_buy_points,
            "next_sell_points": next_sell_points,
            "watch_points": watch_points,
            "recommended_buy_price_range": price_plan["recommended_buy_price_range"],
            "recommended_sell_price_range": price_plan["recommended_sell_price_range"],
            "watch_price_range": price_plan["watch_price_range"],
            "risk_controls": risk_controls,
            "position_advice": position_advice,
            "confidence": 63,
            "disclaimer": "该分析仅供复盘参考，不构成投资建议。",
        }
