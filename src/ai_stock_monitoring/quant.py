from __future__ import annotations

"""Lightweight quant-style ensemble scoring.

设计目标：
- 不承诺真实收益率，只给出统一的技术面/股息面综合评分
- 使用当前项目已经具备的数据（价格、均线、股息率）完成多模型投票
- 参考高星量化项目常见思路：多因子打分、趋势确认、支撑阻力、风险收益比过滤
- 暴露几组实用参数，让用户可以用更严格的趋势 / 波动 / 股息过滤条件提升命中质量
"""

from dataclasses import dataclass
import json
import math
from typing import Iterable, Mapping

from .providers.base import PriceBar


DEFAULT_QUANT_MODELS: tuple[str, ...] = (
    "trend_following",
    "mean_reversion",
    "dividend_quality",
    "weekly_resonance",
    "volatility_filter",
    "support_strength",
    "risk_reward",
)

MODEL_LABELS = {
    "trend_following": "趋势跟随",
    "mean_reversion": "均值回归",
    "dividend_quality": "股息质量",
    "weekly_resonance": "周线共振",
    "volatility_filter": "波动过滤",
    "support_strength": "支撑强度",
    "risk_reward": "盈亏比",
}

DEFAULT_QUANT_STRATEGY_PARAMS: dict[str, float | bool] = {
    "require_price_above_ma250": True,
    "require_weekly_bullish": True,
    "min_dividend_yield": 3.0,
    "max_20d_volatility": 0.04,
    "min_20d_momentum_pct": 0.01,
    "max_boll_deviation_pct": 0.04,
    "support_zone_tolerance_pct": 0.03,
    "min_reward_risk_ratio": 1.6,
}


@dataclass(frozen=True)
class QuantModelResult:
    key: str
    label: str
    score: float
    reason: str


@dataclass(frozen=True)
class QuantSignal:
    probability: float
    summary: str
    breakdown_json: str
    models: tuple[QuantModelResult, ...]


def normalize_selected_models(selected_models: Iterable[str] | None) -> tuple[str, ...]:
    normalized = [item for item in (selected_models or []) if item in MODEL_LABELS]
    return tuple(dict.fromkeys(normalized)) or DEFAULT_QUANT_MODELS


def available_quant_models() -> list[dict[str, str]]:
    return [{"key": key, "label": label} for key, label in MODEL_LABELS.items()]


def normalize_strategy_params(raw_params: Mapping[str, object] | None) -> dict[str, float | bool]:
    params = dict(DEFAULT_QUANT_STRATEGY_PARAMS)
    if raw_params:
        if "require_price_above_ma250" in raw_params:
            params["require_price_above_ma250"] = bool(raw_params["require_price_above_ma250"])
        if "require_weekly_bullish" in raw_params:
            params["require_weekly_bullish"] = bool(raw_params["require_weekly_bullish"])
        if "min_dividend_yield" in raw_params:
            params["min_dividend_yield"] = max(0.0, float(raw_params["min_dividend_yield"]))
        if "max_20d_volatility" in raw_params:
            params["max_20d_volatility"] = max(0.005, float(raw_params["max_20d_volatility"]))
        if "min_20d_momentum_pct" in raw_params:
            params["min_20d_momentum_pct"] = float(raw_params["min_20d_momentum_pct"])
        if "max_boll_deviation_pct" in raw_params:
            params["max_boll_deviation_pct"] = max(0.01, float(raw_params["max_boll_deviation_pct"]))
        if "support_zone_tolerance_pct" in raw_params:
            params["support_zone_tolerance_pct"] = max(0.005, float(raw_params["support_zone_tolerance_pct"]))
        if "min_reward_risk_ratio" in raw_params:
            params["min_reward_risk_ratio"] = max(0.5, float(raw_params["min_reward_risk_ratio"]))
    return params


def build_quant_signal(
    latest_price: float,
    ma_250: float,
    boll_mid: float,
    boll_lower: float,
    boll_upper: float,
    ma_30w: float,
    ma_60w: float,
    dividend_yield: float,
    daily_bars: list[PriceBar],
    weekly_bars: list[PriceBar],
    selected_models: Iterable[str] | None = None,
    strategy_params: Mapping[str, object] | None = None,
) -> QuantSignal:
    """Compute an ensemble probability-like score from several simple models."""

    params = normalize_strategy_params(strategy_params)
    closes = [float(item.close_price) for item in daily_bars]
    weekly_closes = [float(item.close_price) for item in weekly_bars]
    daily_returns = _daily_returns(closes)
    volatility_20 = _annualized_volatility(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    momentum_20_pct = _price_change_ratio(closes[-20], closes[-1]) if len(closes) >= 20 else 0.0

    models: list[QuantModelResult] = []

    for model_key in normalize_selected_models(selected_models):
        if model_key == "trend_following":
            models.append(_trend_following_model(latest_price, ma_250, closes, params))
        elif model_key == "mean_reversion":
            models.append(_mean_reversion_model(latest_price, ma_250, boll_mid, params))
        elif model_key == "dividend_quality":
            models.append(_dividend_quality_model(latest_price, ma_250, dividend_yield, params))
        elif model_key == "weekly_resonance":
            models.append(_weekly_resonance_model(ma_30w, ma_60w, weekly_closes, params))
        elif model_key == "volatility_filter":
            models.append(_volatility_filter_model(volatility_20, momentum_20_pct, params))
        elif model_key == "support_strength":
            models.append(_support_strength_model(latest_price, ma_250, boll_mid, boll_lower, params))
        elif model_key == "risk_reward":
            models.append(_risk_reward_model(latest_price, ma_250, boll_mid, boll_lower, boll_upper, params))

    probability = round(sum(item.score for item in models) / max(len(models), 1), 2)
    summary = _build_summary(probability, models, params)
    breakdown = json.dumps(
        [
            {
                "key": item.key,
                "label": item.label,
                "score": item.score,
                "reason": item.reason,
            }
            for item in models
        ],
        ensure_ascii=False,
    )
    return QuantSignal(
        probability=probability,
        summary=summary,
        breakdown_json=breakdown,
        models=tuple(models),
    )


def _trend_following_model(
    latest_price: float,
    ma_250: float,
    closes: list[float],
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    score = 45.0
    reasons: list[str] = []
    require_above_ma250 = bool(params["require_price_above_ma250"])
    min_20d_momentum_pct = float(params["min_20d_momentum_pct"])

    if ma_250 > 0 and latest_price >= ma_250:
        score += 22
        reasons.append("价格站上 250 日线")
    elif ma_250 > 0:
        score -= 24 if require_above_ma250 else 14
        reasons.append("价格仍在 250 日线下方")

    if len(closes) >= 20:
        momentum_20_pct = _price_change_ratio(closes[-20], closes[-1])
        if momentum_20_pct >= min_20d_momentum_pct:
            score += 18
            reasons.append(f"20 日涨幅达到 {momentum_20_pct * 100:.2f}%")
        else:
            score -= 16
            reasons.append(f"20 日涨幅仅 {momentum_20_pct * 100:.2f}%")

    if len(closes) >= 5:
        slope_5 = _price_change_ratio(closes[-5], closes[-1])
        if slope_5 > 0:
            score += 8
            reasons.append("近 5 日短趋势继续向上")
        else:
            score -= 6
            reasons.append("近 5 日短趋势未走强")

    return QuantModelResult(
        key="trend_following",
        label=MODEL_LABELS["trend_following"],
        score=_clamp_score(score),
        reason="；".join(reasons) or "缺少足够趋势数据",
    )


def _mean_reversion_model(
    latest_price: float,
    ma_250: float,
    boll_mid: float,
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    score = 50.0
    reasons: list[str] = []
    max_boll_deviation_pct = float(params["max_boll_deviation_pct"])
    if boll_mid > 0:
        deviation = (latest_price - boll_mid) / boll_mid
        if abs(deviation) <= max_boll_deviation_pct:
            score += 20
            reasons.append(f"价格距离 BOLL 中轨不超过 {max_boll_deviation_pct * 100:.2f}%")
        elif deviation < -(max_boll_deviation_pct * 2):
            score -= 18
            reasons.append("价格明显跌破中轨，弱势回撤风险更高")
        else:
            score -= 6
            reasons.append("价格偏离 BOLL 中轨较多")
    if ma_250 > 0 and latest_price >= ma_250:
        score += 12
        reasons.append("均值回归发生在长线支撑之上")
    return QuantModelResult(
        key="mean_reversion",
        label=MODEL_LABELS["mean_reversion"],
        score=_clamp_score(score),
        reason="；".join(reasons) or "缺少均值参考数据",
    )


def _dividend_quality_model(
    latest_price: float,
    ma_250: float,
    dividend_yield: float,
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    score = 40.0
    reasons: list[str] = []
    min_dividend_yield = float(params["min_dividend_yield"])
    if dividend_yield >= max(5.0, min_dividend_yield):
        score += 28
        reasons.append("股息率较高")
    elif dividend_yield >= min_dividend_yield:
        score += 18
        reasons.append(f"股息率达到自定义阈值 {min_dividend_yield:.2f}%")
    else:
        score -= 8
        reasons.append(f"股息率低于阈值 {min_dividend_yield:.2f}%")
    if ma_250 > 0 and latest_price >= ma_250:
        score += 12
        reasons.append("高股息同时保持中长期趋势")
    elif ma_250 > 0:
        score -= 8
        reasons.append("高股息但价格仍弱于长线趋势")
    return QuantModelResult(
        key="dividend_quality",
        label=MODEL_LABELS["dividend_quality"],
        score=_clamp_score(score),
        reason="；".join(reasons),
    )


def _weekly_resonance_model(
    ma_30w: float,
    ma_60w: float,
    weekly_closes: list[float],
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    score = 48.0
    reasons: list[str] = []
    require_weekly_bullish = bool(params["require_weekly_bullish"])
    if ma_30w > 0 and ma_60w > 0:
        if ma_30w >= ma_60w:
            score += 22
            reasons.append("30 周均线位于 60 周均线之上")
        else:
            score -= 18 if require_weekly_bullish else 10
            reasons.append("30 周均线仍弱于 60 周均线")
    if len(weekly_closes) >= 4:
        weekly_momentum = weekly_closes[-1] - weekly_closes[-4]
        if weekly_momentum > 0:
            score += 12
            reasons.append("近 4 周趋势继续向上")
        else:
            score -= 6
            reasons.append("近 4 周趋势偏平或回落")
    return QuantModelResult(
        key="weekly_resonance",
        label=MODEL_LABELS["weekly_resonance"],
        score=_clamp_score(score),
        reason="；".join(reasons) or "缺少周线数据",
    )


def _volatility_filter_model(
    volatility_20: float,
    momentum_20_pct: float,
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    score = 52.0
    reasons: list[str] = []
    max_20d_volatility = float(params["max_20d_volatility"])
    if volatility_20 <= max_20d_volatility:
        score += 18
        reasons.append(f"20 日波动率 {volatility_20 * 100:.2f}% 低于阈值")
    else:
        score -= 16
        reasons.append(f"20 日波动率 {volatility_20 * 100:.2f}% 高于阈值 {max_20d_volatility * 100:.2f}%")
    if momentum_20_pct > 0:
        score += 8
        reasons.append("低波动下仍保持正收益")
    else:
        score -= 6
        reasons.append("低波动过滤未配合正趋势")
    return QuantModelResult(
        key="volatility_filter",
        label=MODEL_LABELS["volatility_filter"],
        score=_clamp_score(score),
        reason="；".join(reasons) or "缺少波动率数据",
    )


def _support_strength_model(
    latest_price: float,
    ma_250: float,
    boll_mid: float,
    boll_lower: float,
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    """Evaluate whether price is close to a valid support zone instead of far away from it."""

    score = 50.0
    reasons: list[str] = []
    tolerance = float(params["support_zone_tolerance_pct"])
    support_levels = [value for value in (boll_lower, ma_250, boll_mid) if value > 0]
    if not support_levels:
        return QuantModelResult(
            key="support_strength",
            label=MODEL_LABELS["support_strength"],
            score=_clamp_score(score),
            reason="缺少支撑位数据",
        )

    nearest_support = min(support_levels, key=lambda item: abs(latest_price - item))
    deviation = (latest_price - nearest_support) / nearest_support if nearest_support > 0 else 0.0
    if 0 <= deviation <= tolerance:
        score += 24
        reasons.append(f"价格贴近支撑位 {nearest_support:.2f} 且未跌破")
    elif 0 <= deviation <= tolerance * 2:
        score += 10
        reasons.append(f"价格位于支撑位 {nearest_support:.2f} 上方不远处")
    elif deviation < 0 and abs(deviation) <= tolerance:
        score -= 8
        reasons.append(f"价格小幅跌破支撑位 {nearest_support:.2f}，需要确认是否假跌破")
    else:
        score -= 18
        reasons.append("价格距离主要支撑位偏远，性价比一般")

    if boll_lower > 0 and latest_price < boll_lower:
        score -= 10
        reasons.append("价格跌破 BOLL 下轨，短线承压")

    return QuantModelResult(
        key="support_strength",
        label=MODEL_LABELS["support_strength"],
        score=_clamp_score(score),
        reason="；".join(reasons),
    )


def _risk_reward_model(
    latest_price: float,
    ma_250: float,
    boll_mid: float,
    boll_lower: float,
    boll_upper: float,
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    """Approximate reward/risk using nearby support and resistance.

    这里不做复杂回测，只用当前已有的 BOLL / 均线数据粗略估算：
    - 上方空间：距离最近有效压力位（优先 BOLL 上轨）
    - 下方风险：距离最近有效支撑位（优先 BOLL 下轨 / 250 日线 / 中轨）
    这样能把“看起来不错”进一步收敛成“盈亏比是否划算”。
    """

    score = 50.0
    reasons: list[str] = []
    min_ratio = float(params["min_reward_risk_ratio"])

    support_levels = [value for value in (boll_lower, ma_250, boll_mid) if 0 < value < latest_price]
    support = max(support_levels) if support_levels else 0.0
    resistance_levels = [value for value in (boll_upper,) if value > latest_price]
    resistance = min(resistance_levels) if resistance_levels else 0.0

    if support <= 0 or resistance <= 0:
        if support <= 0:
            score -= 8
            reasons.append("缺少有效下方支撑，止损锚点不清晰")
        if resistance <= 0:
            score -= 8
            reasons.append("上方空间有限或已接近压力位")
        return QuantModelResult(
            key="risk_reward",
            label=MODEL_LABELS["risk_reward"],
            score=_clamp_score(score),
            reason="；".join(reasons),
        )

    downside_pct = (latest_price - support) / latest_price
    upside_pct = (resistance - latest_price) / latest_price
    ratio = upside_pct / max(downside_pct, 0.003)

    if ratio >= min_ratio:
        score += 24
        reasons.append(f"预估盈亏比约 {ratio:.2f}，高于阈值 {min_ratio:.2f}")
    elif ratio >= 1.0:
        score += 10
        reasons.append(f"预估盈亏比约 {ratio:.2f}，尚可但不够优秀")
    else:
        score -= 18
        reasons.append(f"预估盈亏比约 {ratio:.2f}，收益空间不足以覆盖风险")

    if boll_upper > 0 and latest_price >= boll_upper:
        score -= 10
        reasons.append("价格已接近或触及 BOLL 上轨，追高性价比下降")

    return QuantModelResult(
        key="risk_reward",
        label=MODEL_LABELS["risk_reward"],
        score=_clamp_score(score),
        reason="；".join(reasons),
    )


def _daily_returns(closes: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        if previous <= 0:
            continue
        returns.append((current - previous) / previous)
    return returns


def _annualized_volatility(returns: list[float]) -> float:
    if not returns:
        return 0.0
    mean_value = sum(returns) / len(returns)
    variance = sum((item - mean_value) ** 2 for item in returns) / len(returns)
    return math.sqrt(variance) * math.sqrt(252)


def _price_change_ratio(base_price: float, latest_price: float) -> float:
    if base_price <= 0:
        return 0.0
    return (latest_price - base_price) / base_price


def _build_summary(
    probability: float,
    models: list[QuantModelResult],
    params: Mapping[str, float | bool],
) -> str:
    ranked_models = sorted(models, key=lambda item: item.score, reverse=True)
    top_model = ranked_models[0] if ranked_models else None
    secondary_model = ranked_models[1] if len(ranked_models) > 1 else None
    if probability >= 90:
        prefix = "综合评分极强"
    elif probability >= 80:
        prefix = "综合评分偏强"
    elif probability >= 65:
        prefix = "综合评分中性偏强"
    else:
        prefix = "综合评分一般"
    if top_model is None:
        return prefix

    summary = (
        f"{prefix}，当前最强信号来自{top_model.label}模型"
        f"（次强为{secondary_model.label}）" if secondary_model else f"{prefix}，当前最强信号来自{top_model.label}模型"
    )
    return (
        f"{summary}；最小股息率阈值 {float(params['min_dividend_yield']):.2f}% ，"
        f"20 日动量阈值 {float(params['min_20d_momentum_pct']) * 100:.2f}% ，"
        f"20 日波动率上限 {float(params['max_20d_volatility']) * 100:.2f}% ，"
        f"BOLL 偏离上限 {float(params['max_boll_deviation_pct']) * 100:.2f}% ，"
        f"支撑区容忍度 {float(params['support_zone_tolerance_pct']) * 100:.2f}% ，"
        f"最小盈亏比 {float(params['min_reward_risk_ratio']):.2f}。"
    )


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(99.0, score)), 2)
