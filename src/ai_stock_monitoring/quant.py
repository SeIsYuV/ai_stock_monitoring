from __future__ import annotations

"""Lightweight quant-style ensemble scoring.

设计目标：
- 不承诺真实收益率，只给出统一的技术面/股息面综合评分
- 使用当前项目已经具备的数据（价格、均线、股息率）完成多模型投票
- 输出结构尽量稳定，便于页面展示和告警持久化
"""

from dataclasses import dataclass
import json
from typing import Iterable

from .providers.base import PriceBar


DEFAULT_QUANT_MODELS: tuple[str, ...] = (
    "trend_following",
    "mean_reversion",
    "dividend_quality",
    "weekly_resonance",
)

MODEL_LABELS = {
    "trend_following": "趋势跟随",
    "mean_reversion": "均值回归",
    "dividend_quality": "股息质量",
    "weekly_resonance": "周线共振",
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


def build_quant_signal(
    latest_price: float,
    ma_250: float,
    boll_mid: float,
    ma_30w: float,
    ma_60w: float,
    dividend_yield: float,
    daily_bars: list[PriceBar],
    weekly_bars: list[PriceBar],
    selected_models: Iterable[str] | None = None,
) -> QuantSignal:
    """Compute an ensemble probability-like score from several simple models.

    这是“盈利概率评分”，不是回测后的真实胜率承诺。
    页面上会明确展示这是模型综合评分。
    """

    closes = [float(item.close_price) for item in daily_bars]
    weekly_closes = [float(item.close_price) for item in weekly_bars]
    models: list[QuantModelResult] = []

    for model_key in normalize_selected_models(selected_models):
        if model_key == "trend_following":
            models.append(_trend_following_model(latest_price, ma_250, closes))
        elif model_key == "mean_reversion":
            models.append(_mean_reversion_model(latest_price, ma_250, boll_mid))
        elif model_key == "dividend_quality":
            models.append(_dividend_quality_model(latest_price, ma_250, dividend_yield))
        elif model_key == "weekly_resonance":
            models.append(_weekly_resonance_model(ma_30w, ma_60w, weekly_closes))

    probability = round(sum(item.score for item in models) / max(len(models), 1), 2)
    summary = _build_summary(probability, models)
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


def _trend_following_model(latest_price: float, ma_250: float, closes: list[float]) -> QuantModelResult:
    score = 45.0
    reasons: list[str] = []
    if ma_250 > 0 and latest_price >= ma_250:
        score += 22
        reasons.append("价格站上 250 日线")
    elif ma_250 > 0:
        score -= 18
        reasons.append("价格仍在 250 日线下方")

    if len(closes) >= 20:
        momentum_20 = closes[-1] - closes[-20]
        if momentum_20 > 0:
            score += 18
            reasons.append("近 20 日动量为正")
        else:
            score -= 10
            reasons.append("近 20 日动量偏弱")
    if len(closes) >= 5:
        momentum_5 = closes[-1] - closes[-5]
        if momentum_5 > 0:
            score += 8
            reasons.append("短线继续抬升")
        else:
            score -= 5
            reasons.append("短线延续性一般")
    return QuantModelResult(
        key="trend_following",
        label=MODEL_LABELS["trend_following"],
        score=_clamp_score(score),
        reason="；".join(reasons) or "缺少足够趋势数据",
    )


def _mean_reversion_model(latest_price: float, ma_250: float, boll_mid: float) -> QuantModelResult:
    score = 50.0
    reasons: list[str] = []
    if boll_mid > 0:
        deviation = (latest_price - boll_mid) / boll_mid
        if -0.03 <= deviation <= 0.02:
            score += 20
            reasons.append("价格靠近 BOLL 中轨，回撤后再上攻的性价比较好")
        elif deviation < -0.08:
            score -= 18
            reasons.append("价格明显跌破中轨，弱势回撤风险更高")
        else:
            score -= 4
            reasons.append("价格相对中轨不够理想")
    if ma_250 > 0 and latest_price >= ma_250:
        score += 12
        reasons.append("均值回归发生在长线支撑之上")
    return QuantModelResult(
        key="mean_reversion",
        label=MODEL_LABELS["mean_reversion"],
        score=_clamp_score(score),
        reason="；".join(reasons) or "缺少均值参考数据",
    )


def _dividend_quality_model(latest_price: float, ma_250: float, dividend_yield: float) -> QuantModelResult:
    score = 40.0
    reasons: list[str] = []
    if dividend_yield >= 5.0:
        score += 28
        reasons.append("股息率较高")
    elif dividend_yield >= 4.0:
        score += 18
        reasons.append("股息率达到观察区间")
    else:
        score += 5
        reasons.append("股息率一般")
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


def _weekly_resonance_model(ma_30w: float, ma_60w: float, weekly_closes: list[float]) -> QuantModelResult:
    score = 48.0
    reasons: list[str] = []
    if ma_30w > 0 and ma_60w > 0:
        if ma_30w >= ma_60w:
            score += 22
            reasons.append("30 周均线位于 60 周均线之上")
        else:
            score -= 14
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


def _build_summary(probability: float, models: list[QuantModelResult]) -> str:
    top_model = max(models, key=lambda item: item.score) if models else None
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
    return f"{prefix}，当前最强信号来自{top_model.label}模型。"


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(99.0, score)), 2)
