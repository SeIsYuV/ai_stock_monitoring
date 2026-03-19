from __future__ import annotations

"""Lightweight quant-style ensemble scoring.

设计目标：
- 不承诺真实收益率，只给出统一的技术面/股息面综合评分
- 使用当前项目已经具备的数据（价格、均线、股息率）完成多模型投票
- 参考高星量化项目常见思路：多因子打分、趋势确认、支撑阻力、风险收益比过滤
- 暴露几组实用参数，让用户可以用更严格的趋势 / 波动 / 股息过滤条件提升命中质量
"""

from dataclasses import dataclass, replace
import json
import math
from typing import Any, Iterable, Mapping

from .providers.base import PriceBar


DEFAULT_QUANT_MODELS: tuple[str, ...] = (
    "trend_following",
    "mean_reversion",
    "dividend_quality",
    "weekly_resonance",
    "volatility_filter",
    "support_strength",
    "risk_reward",
    "dcf_proxy",
    "msci_momentum",
    "quality_stability",
)

MODEL_LABELS = {
    "trend_following": "趋势跟随",
    "mean_reversion": "均值回归",
    "dividend_quality": "股息质量",
    "weekly_resonance": "周线共振",
    "volatility_filter": "波动过滤",
    "support_strength": "支撑强度",
    "risk_reward": "盈亏比",
    "dcf_proxy": "DCF估值",
    "msci_momentum": "MSCI动量",
    "quality_stability": "质量稳定",
}

PROFESSIONAL_MODEL_KEYS = {"msci_momentum", "quality_stability"}

DEFAULT_QUANT_STRATEGY_PARAMS: dict[str, float | bool] = {
    "require_price_above_ma250": True,
    "require_weekly_bullish": True,
    "min_dividend_yield": 3.0,
    "max_20d_volatility": 0.04,
    "min_20d_momentum_pct": 0.01,
    "max_boll_deviation_pct": 0.04,
    "support_zone_tolerance_pct": 0.03,
    "min_reward_risk_ratio": 1.6,
    "dcf_discount_rate": 0.10,
    "dcf_terminal_growth": 0.03,
    "adaptive_learning_enabled": True,
    "adaptive_lookback_days": 180,
    "adaptive_holding_days": 10,
    "adaptive_min_samples": 12,
    "adaptive_target_return_pct": 0.03,
    "adaptive_recent_window_days": 45,
    "adaptive_recent_emphasis": 0.65,
    "adaptive_stability_penalty": 0.35,
}


@dataclass(frozen=True)
class QuantModelResult:
    key: str
    label: str
    score: float
    reason: str
    intrinsic_value: float | None = None
    valuation_gap_pct: float | None = None
    base_score: float | None = None
    adaptive_weight: float = 1.0
    adaptive_sample_size: int = 0
    adaptive_hit_rate: float | None = None
    adaptive_avg_return_pct: float | None = None


@dataclass(frozen=True)
class QuantSignal:
    probability: float
    summary: str
    breakdown_json: str
    models: tuple[QuantModelResult, ...]


@dataclass(frozen=True)
class AdaptiveLearningProfile:
    weight: float = 1.0
    sample_size: int = 0
    hit_rate: float | None = None
    avg_return_pct: float | None = None


ADAPTIVE_SIGNAL_SCORE_THRESHOLD = 60.0


@dataclass(frozen=True)
class EnsembleLearningProfiles:
    model_profiles: dict[str, AdaptiveLearningProfile]
    group_profiles: dict[str, AdaptiveLearningProfile]


def normalize_selected_models(selected_models: Iterable[str] | None) -> tuple[str, ...]:
    normalized = [item for item in (selected_models or []) if item in MODEL_LABELS]
    return tuple(dict.fromkeys(normalized)) or DEFAULT_QUANT_MODELS


def available_quant_models() -> list[dict[str, str]]:
    return [{"key": key, "label": label} for key, label in MODEL_LABELS.items()]


def extract_dcf_metrics(quant_model_breakdown: str | None) -> dict[str, float | None]:
    try:
        breakdown = json.loads(str(quant_model_breakdown or "[]"))
    except json.JSONDecodeError:
        breakdown = []
    for item in breakdown:
        if str(item.get("key") or "") == "dcf_proxy":
            intrinsic_value = item.get("intrinsic_value")
            valuation_gap_pct = item.get("valuation_gap_pct")
            return {
                "dcf_intrinsic_value": round(float(intrinsic_value), 2) if intrinsic_value is not None else None,
                "dcf_valuation_gap_pct": round(float(valuation_gap_pct), 2) if valuation_gap_pct is not None else None,
                "dcf_label": str(item.get("label") or MODEL_LABELS["dcf_proxy"]),
                "dcf_reason": str(item.get("reason") or ""),
            }
    return {
        "dcf_intrinsic_value": None,
        "dcf_valuation_gap_pct": None,
        "dcf_label": MODEL_LABELS["dcf_proxy"],
        "dcf_reason": "未启用 DCF 模型或当前估值数据不足",
    }


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
        if "dcf_discount_rate" in raw_params:
            params["dcf_discount_rate"] = max(0.05, float(raw_params["dcf_discount_rate"]))
        if "dcf_terminal_growth" in raw_params:
            params["dcf_terminal_growth"] = max(0.0, min(0.08, float(raw_params["dcf_terminal_growth"])))
        if "adaptive_learning_enabled" in raw_params:
            params["adaptive_learning_enabled"] = bool(raw_params["adaptive_learning_enabled"])
        if "adaptive_lookback_days" in raw_params:
            params["adaptive_lookback_days"] = max(60, min(360, int(float(raw_params["adaptive_lookback_days"]))))
        if "adaptive_holding_days" in raw_params:
            params["adaptive_holding_days"] = max(3, min(30, int(float(raw_params["adaptive_holding_days"]))))
        if "adaptive_min_samples" in raw_params:
            params["adaptive_min_samples"] = max(4, min(80, int(float(raw_params["adaptive_min_samples"]))))
        if "adaptive_target_return_pct" in raw_params:
            params["adaptive_target_return_pct"] = max(0.0, min(0.20, float(raw_params["adaptive_target_return_pct"])))
        if "adaptive_recent_window_days" in raw_params:
            params["adaptive_recent_window_days"] = max(10, min(120, int(float(raw_params["adaptive_recent_window_days"]))))
        if "adaptive_recent_emphasis" in raw_params:
            params["adaptive_recent_emphasis"] = max(0.1, min(0.9, float(raw_params["adaptive_recent_emphasis"])))
        if "adaptive_stability_penalty" in raw_params:
            params["adaptive_stability_penalty"] = max(0.0, min(1.0, float(raw_params["adaptive_stability_penalty"])))
    if float(params["dcf_terminal_growth"]) >= float(params["dcf_discount_rate"]) - 0.01:
        params["dcf_terminal_growth"] = round(float(params["dcf_discount_rate"]) - 0.02, 4)
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
    symbol_fundamentals: Mapping[str, float | None] | None = None,
    live_feedback: Mapping[str, Mapping[str, float | int]] | None = None,
    selected_models: Iterable[str] | None = None,
    strategy_params: Mapping[str, object] | None = None,
) -> QuantSignal:
    """Compute an ensemble probability-like score from several simple models."""

    params = normalize_strategy_params(strategy_params)
    closes = [float(item.close_price) for item in daily_bars]
    weekly_closes = [float(item.close_price) for item in weekly_bars]
    model_keys = normalize_selected_models(selected_models)
    models = [
        _build_model_result(
            model_key,
            latest_price=latest_price,
            ma_250=ma_250,
            boll_mid=boll_mid,
            boll_lower=boll_lower,
            boll_upper=boll_upper,
            ma_30w=ma_30w,
            ma_60w=ma_60w,
            dividend_yield=dividend_yield,
            closes=closes,
            weekly_closes=weekly_closes,
            symbol_fundamentals=symbol_fundamentals or {},
            params=params,
        )
        for model_key in model_keys
    ]

    learning_profiles = EnsembleLearningProfiles(model_profiles={}, group_profiles={})
    if bool(params["adaptive_learning_enabled"]):
        learning_profiles = _simulate_adaptive_learning(
            daily_bars=daily_bars,
            weekly_bars=weekly_bars,
            dividend_yield=dividend_yield,
            symbol_fundamentals=symbol_fundamentals or {},
            selected_models=model_keys,
            params=params,
        )
        models = [
            _apply_adaptive_profile(
                model,
                learning_profiles.model_profiles.get(model.key, AdaptiveLearningProfile()),
                holding_days=int(params["adaptive_holding_days"]),
            )
            for model in models
        ]

    feedback_payload = dict(live_feedback or {})
    if feedback_payload:
        models = [
            _apply_live_feedback(
                model,
                feedback_payload.get(model.key),
            )
            for model in models
        ]

    probability = round(
        _blend_group_scores(
            models,
            learning_profiles.group_profiles,
            feedback_payload,
        ),
        2,
    )
    summary = _build_summary(probability, models, params, learning_profiles.group_profiles, feedback_payload)
    breakdown = json.dumps(
        [
            {
                "key": item.key,
                "label": item.label,
                "score": item.score,
                "reason": item.reason,
                "intrinsic_value": item.intrinsic_value,
                "valuation_gap_pct": item.valuation_gap_pct,
                "base_score": item.base_score,
                "adaptive_weight": item.adaptive_weight,
                "adaptive_sample_size": item.adaptive_sample_size,
                "adaptive_hit_rate": item.adaptive_hit_rate,
                "adaptive_avg_return_pct": item.adaptive_avg_return_pct,
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


def _build_model_result(
    model_key: str,
    *,
    latest_price: float,
    ma_250: float,
    boll_mid: float,
    boll_lower: float,
    boll_upper: float,
    ma_30w: float,
    ma_60w: float,
    dividend_yield: float,
    closes: list[float],
    weekly_closes: list[float],
    symbol_fundamentals: Mapping[str, float | None],
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    daily_returns = _daily_returns(closes)
    volatility_20 = _annualized_volatility(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    momentum_20_pct = _price_change_ratio(closes[-20], closes[-1]) if len(closes) >= 20 else 0.0

    if model_key == "trend_following":
        return _trend_following_model(latest_price, ma_250, closes, params)
    if model_key == "mean_reversion":
        return _mean_reversion_model(latest_price, ma_250, boll_mid, params)
    if model_key == "dividend_quality":
        return _dividend_quality_model(latest_price, ma_250, dividend_yield, params)
    if model_key == "weekly_resonance":
        return _weekly_resonance_model(ma_30w, ma_60w, weekly_closes, params)
    if model_key == "volatility_filter":
        return _volatility_filter_model(volatility_20, momentum_20_pct, params)
    if model_key == "support_strength":
        return _support_strength_model(latest_price, ma_250, boll_mid, boll_lower, params)
    if model_key == "risk_reward":
        return _risk_reward_model(latest_price, ma_250, boll_mid, boll_lower, boll_upper, params)
    if model_key == "dcf_proxy":
        return _dcf_proxy_model(latest_price, dividend_yield, closes, weekly_closes, params)
    if model_key == "msci_momentum":
        return _msci_momentum_model(latest_price, ma_250, closes, params)
    if model_key == "quality_stability":
        return _quality_stability_model(latest_price, ma_250, dividend_yield, closes, weekly_closes, symbol_fundamentals, params)
    raise ValueError(f"Unsupported quant model: {model_key}")


def _simulate_adaptive_learning(
    daily_bars: list[PriceBar],
    weekly_bars: list[PriceBar],
    dividend_yield: float,
    symbol_fundamentals: Mapping[str, float | None],
    selected_models: tuple[str, ...],
    params: Mapping[str, float | bool],
) -> EnsembleLearningProfiles:
    holding_days = int(params["adaptive_holding_days"])
    lookback_days = int(params["adaptive_lookback_days"])
    min_samples = int(params["adaptive_min_samples"])
    target_return = float(params["adaptive_target_return_pct"])
    recent_window_days = int(params["adaptive_recent_window_days"])
    recent_emphasis = float(params["adaptive_recent_emphasis"])
    stability_penalty = float(params["adaptive_stability_penalty"])
    if len(daily_bars) <= holding_days + 25:
        return EnsembleLearningProfiles(model_profiles={}, group_profiles={})

    lookback_span = max(holding_days + 25, lookback_days + holding_days)
    recent_daily_bars = daily_bars[-lookback_span:]
    samples_by_model: dict[str, list[float]] = {key: [] for key in selected_models}
    group_samples: dict[str, list[float]] = {"professional": [], "adaptive": []}

    for index in range(20, len(recent_daily_bars) - holding_days):
        history_daily = recent_daily_bars[: index + 1]
        current_date = history_daily[-1].traded_on
        history_weekly = [item for item in weekly_bars if item.traded_on <= current_date]
        if len(history_daily) < 20 or not history_weekly:
            continue
        closes = [float(item.close_price) for item in history_daily]
        weekly_closes = [float(item.close_price) for item in history_weekly]
        latest_price = closes[-1]
        ma_250 = _simple_moving_average(closes, 250)
        boll_mid = _simple_moving_average(closes, 20)
        boll_lower = _bollinger_lower_band(closes, 20)
        boll_upper = _bollinger_upper_band(closes, 20)
        ma_30w = _simple_moving_average(weekly_closes, 30)
        ma_60w = _simple_moving_average(weekly_closes, 60)
        forward_price = float(recent_daily_bars[index + holding_days].close_price)
        if latest_price <= 0:
            continue
        forward_return = (forward_price - latest_price) / latest_price

        iteration_results: dict[str, QuantModelResult] = {}
        for model_key in selected_models:
            simulated = _build_model_result(
                model_key,
                latest_price=latest_price,
                ma_250=ma_250,
                boll_mid=boll_mid,
                boll_lower=boll_lower,
                boll_upper=boll_upper,
                ma_30w=ma_30w,
                ma_60w=ma_60w,
                dividend_yield=dividend_yield,
                closes=closes,
                weekly_closes=weekly_closes,
                symbol_fundamentals=symbol_fundamentals,
                params=params,
            )
            iteration_results[model_key] = simulated
            if simulated.score >= ADAPTIVE_SIGNAL_SCORE_THRESHOLD:
                samples_by_model[model_key].append(forward_return)

        professional_scores = [
            result.score for key, result in iteration_results.items() if key in PROFESSIONAL_MODEL_KEYS
        ]
        adaptive_scores = [
            result.score for key, result in iteration_results.items() if key not in PROFESSIONAL_MODEL_KEYS
        ]
        if professional_scores and sum(professional_scores) / len(professional_scores) >= ADAPTIVE_SIGNAL_SCORE_THRESHOLD:
            group_samples["professional"].append(forward_return)
        if adaptive_scores and sum(adaptive_scores) / len(adaptive_scores) >= ADAPTIVE_SIGNAL_SCORE_THRESHOLD:
            group_samples["adaptive"].append(forward_return)

    adaptive_profiles: dict[str, AdaptiveLearningProfile] = {}
    for model_key, sample_returns in samples_by_model.items():
        adaptive_profiles[model_key] = _build_learning_profile(
            sample_returns,
            min_samples,
            target_return,
            recent_window_days=recent_window_days,
            recent_emphasis=recent_emphasis,
            stability_penalty=stability_penalty,
        )

    group_profiles = {
        group_key: _build_learning_profile(
            sample_returns,
            min_samples,
            target_return,
            recent_window_days=recent_window_days,
            recent_emphasis=recent_emphasis,
            stability_penalty=stability_penalty,
        )
        for group_key, sample_returns in group_samples.items()
    }
    return EnsembleLearningProfiles(
        model_profiles=adaptive_profiles,
        group_profiles=group_profiles,
    )


def _apply_adaptive_profile(
    model: QuantModelResult,
    profile: AdaptiveLearningProfile,
    holding_days: int,
) -> QuantModelResult:
    base_score = model.score
    if profile.sample_size <= 0:
        return replace(model, base_score=base_score)

    adjusted_score = _clamp_score(base_score * (1.0 + (profile.weight - 1.0) * 0.35))
    learning_reason = (
        f"；滚动模拟 {profile.sample_size} 次，{holding_days} 日目标命中率 {profile.hit_rate:.2f}%"
        f"，平均收益 {profile.avg_return_pct:.2f}%"
    )
    return replace(
        model,
        score=adjusted_score,
        reason=f"{model.reason}{learning_reason}",
        base_score=base_score,
        adaptive_weight=profile.weight,
        adaptive_sample_size=profile.sample_size,
        adaptive_hit_rate=profile.hit_rate,
        adaptive_avg_return_pct=profile.avg_return_pct,
    )


def _build_learning_profile(
    sample_returns: list[float],
    min_samples: int,
    target_return: float,
    *,
    recent_window_days: int,
    recent_emphasis: float,
    stability_penalty: float,
) -> AdaptiveLearningProfile:
    sample_size = len(sample_returns)
    if sample_size < min_samples:
        return AdaptiveLearningProfile(sample_size=sample_size)
    hit_rate = sum(1 for value in sample_returns if value >= target_return) / sample_size
    avg_return = sum(sample_returns) / sample_size
    recent_returns = sample_returns[-max(1, recent_window_days):]
    recent_hit_rate = sum(1 for value in recent_returns if value >= target_return) / max(len(recent_returns), 1)
    recent_avg_return = sum(recent_returns) / max(len(recent_returns), 1)
    blended_hit_rate = hit_rate * (1.0 - recent_emphasis) + recent_hit_rate * recent_emphasis
    blended_avg_return = avg_return * (1.0 - recent_emphasis) + recent_avg_return * recent_emphasis
    variance = sum((value - avg_return) ** 2 for value in sample_returns) / sample_size if sample_size > 1 else 0.0
    return_volatility = math.sqrt(max(variance, 0.0))
    stability_factor = max(0.82, min(1.05, 1.0 - return_volatility * stability_penalty * 4.0))
    confidence = min(1.0, sample_size / max(min_samples * 2, 1))
    raw_weight = 1.0 + (blended_hit_rate - 0.5) * 0.95 + blended_avg_return * 2.8
    raw_weight *= stability_factor
    blended_weight = 1.0 + (raw_weight - 1.0) * confidence
    return AdaptiveLearningProfile(
        weight=round(max(0.75, min(1.35, blended_weight)), 3),
        sample_size=sample_size,
        hit_rate=round(blended_hit_rate * 100, 2),
        avg_return_pct=round(blended_avg_return * 100, 2),
    )


def _apply_live_feedback(
    model: QuantModelResult,
    feedback: Mapping[str, float | int] | None,
) -> QuantModelResult:
    if not feedback:
        return model
    sample_size = int(feedback.get("sample_size") or 0)
    if sample_size < 3:
        return model
    hit_rate = float(feedback.get("hit_rate") or 0.0)
    avg_return_pct = float(feedback.get("avg_return_pct") or 0.0)
    confidence = min(1.0, sample_size / 12.0)
    raw_weight = 1.0 + ((hit_rate / 100.0) - 0.5) * 0.4 + (avg_return_pct / 100.0) * 1.2
    weight = 1.0 + (raw_weight - 1.0) * confidence
    adjusted_score = _clamp_score(model.score * (1.0 + (weight - 1.0) * 0.18))
    return replace(
        model,
        score=adjusted_score,
        reason=(
            f"{model.reason}；纸面交易 {sample_size} 笔，命中率 {hit_rate:.2f}%"
            f"，平均收益 {avg_return_pct:.2f}%"
        ),
    )


def _blend_group_scores(
    models: list[QuantModelResult],
    group_profiles: Mapping[str, AdaptiveLearningProfile],
    live_feedback: Mapping[str, Mapping[str, float | int]] | None = None,
) -> float:
    if not models:
        return 0.0
    professional_scores = [item.score for item in models if item.key in PROFESSIONAL_MODEL_KEYS]
    adaptive_scores = [item.score for item in models if item.key not in PROFESSIONAL_MODEL_KEYS]
    if not professional_scores or not adaptive_scores:
        return sum(item.score for item in models) / len(models)

    professional_avg = sum(professional_scores) / len(professional_scores)
    adaptive_avg = sum(adaptive_scores) / len(adaptive_scores)
    professional_weight = _group_weight(group_profiles.get("professional")) * _feedback_group_weight((live_feedback or {}).get("professional"))
    adaptive_weight = _group_weight(group_profiles.get("adaptive")) * _feedback_group_weight((live_feedback or {}).get("adaptive"))
    total_weight = professional_weight + adaptive_weight
    if total_weight <= 0:
        return sum(item.score for item in models) / len(models)
    return (professional_avg * professional_weight + adaptive_avg * adaptive_weight) / total_weight


def _group_weight(profile: AdaptiveLearningProfile | None) -> float:
    if profile is None or profile.sample_size <= 0:
        return 1.0
    return max(0.85, min(1.35, profile.weight))


def _feedback_group_weight(feedback: Mapping[str, float | int] | None) -> float:
    if not feedback:
        return 1.0
    sample_size = int(feedback.get("sample_size") or 0)
    if sample_size < 3:
        return 1.0
    hit_rate = float(feedback.get("hit_rate") or 0.0)
    avg_return_pct = float(feedback.get("avg_return_pct") or 0.0)
    confidence = min(1.0, sample_size / 12.0)
    raw_weight = 1.0 + ((hit_rate / 100.0) - 0.5) * 0.45 + (avg_return_pct / 100.0) * 1.3
    return max(0.85, min(1.35, 1.0 + (raw_weight - 1.0) * confidence))


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


def _msci_momentum_model(
    latest_price: float,
    ma_250: float,
    closes: list[float],
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    """Approximate MSCI-style medium-term momentum with a 1-month skip and volatility penalty."""

    score = 48.0
    reasons: list[str] = []
    if len(closes) < 147:
        return QuantModelResult(
            key="msci_momentum",
            label=MODEL_LABELS["msci_momentum"],
            score=_clamp_score(score),
            reason="历史样本不足，暂无法计算 6-12 个月动量",
        )

    six_month_skip = _price_change_ratio(closes[-147], closes[-21]) if len(closes) >= 147 else 0.0
    twelve_month_skip = _price_change_ratio(closes[-273], closes[-21]) if len(closes) >= 273 else six_month_skip
    momentum_returns = _daily_returns(closes[-126:]) if len(closes) >= 126 else _daily_returns(closes)
    momentum_vol = _annualized_volatility(momentum_returns[-60:]) if len(momentum_returns) >= 60 else _annualized_volatility(momentum_returns)
    risk_adjusted_momentum = ((six_month_skip * 0.6) + (twelve_month_skip * 0.4)) / max(momentum_vol, 0.08)

    if risk_adjusted_momentum >= 0.35:
        score += 26
        reasons.append("6-12 个月风险调整后动量显著为正")
    elif risk_adjusted_momentum >= 0.12:
        score += 14
        reasons.append("中期风险调整后动量保持正值")
    else:
        score -= 18
        reasons.append("中期动量不足或波动侵蚀收益")

    if ma_250 > 0 and latest_price >= ma_250:
        score += 10
        reasons.append("价格位于 250 日线之上")
    elif ma_250 > 0:
        score -= 10
        reasons.append("价格仍低于 250 日线")

    near_high_ratio = latest_price / max(max(closes[-252:]), 0.01) if len(closes) >= 252 else 0.0
    if near_high_ratio >= 0.9:
        score += 8
        reasons.append("价格维持在近一年高位区附近")
    elif near_high_ratio > 0:
        score -= 4
        reasons.append("价格距离近一年强势区仍有差距")

    return QuantModelResult(
        key="msci_momentum",
        label=MODEL_LABELS["msci_momentum"],
        score=_clamp_score(score),
        reason="；".join(reasons),
    )


def _quality_stability_model(
    latest_price: float,
    ma_250: float,
    dividend_yield: float,
    closes: list[float],
    weekly_closes: list[float],
    symbol_fundamentals: Mapping[str, float | None],
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    """Approximate quality/stability using available valuation plus defensive price behavior."""

    score = 50.0
    reasons: list[str] = []
    pe_ttm = _safe_float(symbol_fundamentals.get("pe_ttm"))
    pb = _safe_float(symbol_fundamentals.get("pb"))
    market_cap = _safe_float(symbol_fundamentals.get("market_cap"))
    volatility_60 = _annualized_volatility(_daily_returns(closes[-60:])) if len(closes) >= 60 else 0.0
    weekly_drawdown = _rolling_drawdown(weekly_closes[-26:]) if len(weekly_closes) >= 26 else 0.0

    if pe_ttm is not None and 0 < pe_ttm <= 28:
        score += 10
        reasons.append(f"市盈率约 {pe_ttm:.2f}，估值不过热")
    elif pe_ttm is not None and pe_ttm > 40:
        score -= 8
        reasons.append(f"市盈率约 {pe_ttm:.2f}，估值偏高")
    else:
        reasons.append("缺少稳定 PE 数据，改用价格稳定性做代理")

    if pb is not None and 0 < pb <= 3.5:
        score += 10
        reasons.append(f"市净率约 {pb:.2f}，资产定价相对克制")
    elif pb is not None and pb > 6:
        score -= 8
        reasons.append(f"市净率约 {pb:.2f}，估值弹性较大")

    if dividend_yield >= 3.0:
        score += 10
        reasons.append("股息率达到质量筛选参考线")
    elif dividend_yield > 0:
        score += 3
        reasons.append("存在现金分红，但防御性一般")
    else:
        score -= 6
        reasons.append("缺少现金分红支撑")

    if volatility_60 and volatility_60 <= 0.28:
        score += 10
        reasons.append(f"近 60 日年化波动约 {volatility_60 * 100:.2f}%，稳定性较好")
    elif volatility_60:
        score -= 10
        reasons.append(f"近 60 日年化波动约 {volatility_60 * 100:.2f}%，稳定性偏弱")

    if weekly_drawdown <= 0.12:
        score += 8
        reasons.append("近半年周线回撤控制较好")
    elif weekly_drawdown >= 0.22:
        score -= 8
        reasons.append("近半年周线回撤偏大")

    if ma_250 > 0 and latest_price >= ma_250:
        score += 6
        reasons.append("长期趋势未破坏")

    if market_cap is not None and market_cap >= 20_000_000_000:
        score += 4
        reasons.append("市值体量较大，波动通常更可控")

    return QuantModelResult(
        key="quality_stability",
        label=MODEL_LABELS["quality_stability"],
        score=_clamp_score(score),
        reason="；".join(reasons),
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


def _dcf_proxy_model(
    latest_price: float,
    dividend_yield: float,
    closes: list[float],
    weekly_closes: list[float],
    params: Mapping[str, float | bool],
) -> QuantModelResult:
    """Approximate a DCF-style valuation using cashflow proxies.

    当前项目的数据源暂无完整财报自由现金流，因此这里采用保守代理：
    - 以每股股息现金流作为当前可观测现金流近似
    - 用日线/周线动量对未来 5 年增长做小幅修正
    - 再用折现率 + 永续增长率估算每股内在价值

    这不是标准财报级 DCF，但可以把“收益型股票是否值得继续持有/加仓”
    融入现有量化组合，作为一个偏基本面方向的辅助因子。
    """

    score = 46.0
    reasons: list[str] = []
    discount_rate = float(params["dcf_discount_rate"])
    terminal_growth = float(params["dcf_terminal_growth"])

    if latest_price <= 0 or dividend_yield <= 0:
        return QuantModelResult(
            key="dcf_proxy",
            label=MODEL_LABELS["dcf_proxy"],
            score=_clamp_score(35.0),
            reason="缺少稳定现金流代理数据，DCF 估值置信度较低",
        )

    annual_cash_flow = latest_price * (dividend_yield / 100)
    momentum_20 = _price_change_ratio(closes[-20], closes[-1]) if len(closes) >= 20 else 0.0
    weekly_momentum = _price_change_ratio(weekly_closes[-8], weekly_closes[-1]) if len(weekly_closes) >= 8 else 0.0
    growth_adjustment = max(-0.02, min(0.03, momentum_20 * 0.5 + weekly_momentum * 0.2))
    projected_growth = max(-0.01, min(discount_rate - 0.02, terminal_growth + growth_adjustment))

    present_value = 0.0
    cash_flow = annual_cash_flow
    for year in range(1, 6):
        cash_flow *= 1 + projected_growth
        present_value += cash_flow / ((1 + discount_rate) ** year)

    terminal_cash_flow = cash_flow * (1 + terminal_growth)
    terminal_value = terminal_cash_flow / max(discount_rate - terminal_growth, 0.01)
    intrinsic_value = present_value + terminal_value / ((1 + discount_rate) ** 5)
    valuation_gap = (intrinsic_value - latest_price) / latest_price

    reasons.append(
        f"DCF 代理估值约 {intrinsic_value:.2f}，相对现价偏差 {valuation_gap * 100:.2f}%"
    )
    reasons.append(
        "当前数据源暂无自由现金流，DCF 暂以股息现金流作保守代理"
    )

    if valuation_gap >= 0.25:
        score += 28
        reasons.append("估值折价较明显")
    elif valuation_gap >= 0.10:
        score += 18
        reasons.append("估值略有安全边际")
    elif valuation_gap >= -0.05:
        score += 6
        reasons.append("估值基本合理")
    else:
        score -= 16
        reasons.append("现价已接近或高于代理内在价值")

    if dividend_yield >= 4.5:
        score += 8
        reasons.append("股息现金流较充足，DCF 参考意义更强")
    elif dividend_yield < 2.5:
        score -= 6
        reasons.append("股息现金流偏弱，DCF 代理稳定性一般")

    return QuantModelResult(
        key="dcf_proxy",
        label=MODEL_LABELS["dcf_proxy"],
        score=_clamp_score(score),
        reason="；".join(reasons),
        intrinsic_value=round(intrinsic_value, 2),
        valuation_gap_pct=round(valuation_gap * 100, 2),
    )


def _daily_returns(closes: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        if previous <= 0:
            continue
        returns.append((current - previous) / previous)
    return returns


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed


def _rolling_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    return max_drawdown


def _simple_moving_average(values: list[float], window: int) -> float:
    if len(values) < window:
        return 0.0
    return round(sum(values[-window:]) / window, 2)


def _bollinger_lower_band(values: list[float], window: int = 20, std_multiplier: float = 2.0) -> float:
    if len(values) < window:
        return 0.0
    sample = values[-window:]
    middle = sum(sample) / window
    variance = sum((item - middle) ** 2 for item in sample) / window
    return round(middle - std_multiplier * math.sqrt(variance), 2)


def _bollinger_upper_band(values: list[float], window: int = 20, std_multiplier: float = 2.0) -> float:
    if len(values) < window:
        return 0.0
    sample = values[-window:]
    middle = sum(sample) / window
    variance = sum((item - middle) ** 2 for item in sample) / window
    return round(middle + std_multiplier * math.sqrt(variance), 2)


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
    group_profiles: Mapping[str, AdaptiveLearningProfile] | None = None,
    live_feedback: Mapping[str, Mapping[str, float | int]] | None = None,
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
    summary = (
        f"{summary}；最小股息率阈值 {float(params['min_dividend_yield']):.2f}% ，"
        f"20 日动量阈值 {float(params['min_20d_momentum_pct']) * 100:.2f}% ，"
        f"20 日波动率上限 {float(params['max_20d_volatility']) * 100:.2f}% ，"
        f"BOLL 偏离上限 {float(params['max_boll_deviation_pct']) * 100:.2f}% ，"
        f"支撑区容忍度 {float(params['support_zone_tolerance_pct']) * 100:.2f}% ，"
        f"最小盈亏比 {float(params['min_reward_risk_ratio']):.2f} ，"
        f"DCF 折现率 {float(params['dcf_discount_rate']) * 100:.2f}% / 永续增长 {float(params['dcf_terminal_growth']) * 100:.2f}%。"
    )
    if bool(params.get("adaptive_learning_enabled")):
        professional_profile = (group_profiles or {}).get("professional")
        adaptive_profile = (group_profiles or {}).get("adaptive")
        learning_clause = (
            f" 自适应层已根据最近 {int(params['adaptive_lookback_days'])} 个交易日的滚动模拟结果，"
            f"按 {int(params['adaptive_holding_days'])} 日持有周期自动校准各子模型权重。"
            f" 其中最近 {int(params['adaptive_recent_window_days'])} 个样本窗口会以"
            f" {float(params['adaptive_recent_emphasis']) * 100:.0f}% 的权重优先参与学习，"
            f"波动过大的模型还会按稳定性系数自动降权。"
        )
        if (
            professional_profile is not None and professional_profile.sample_size > 0
            and adaptive_profile is not None and adaptive_profile.sample_size > 0
        ):
            learning_clause += (
                f" 专业基准组最近命中率 {professional_profile.hit_rate:.2f}% / 平均收益 {professional_profile.avg_return_pct:.2f}% ，"
                f"自适应组最近命中率 {adaptive_profile.hit_rate:.2f}% / 平均收益 {adaptive_profile.avg_return_pct:.2f}% 。"
            )
        professional_live = (live_feedback or {}).get("professional")
        adaptive_live = (live_feedback or {}).get("adaptive")
        if professional_live and adaptive_live:
            learning_clause += (
                f" 真实纸面交易方面，专业组近 {int(professional_live.get('sample_size') or 0)} 笔命中率"
                f" {float(professional_live.get('hit_rate') or 0.0):.2f}% ，"
                f"自适应组近 {int(adaptive_live.get('sample_size') or 0)} 笔命中率"
                f" {float(adaptive_live.get('hit_rate') or 0.0):.2f}% 。"
            )
        return f"{summary}{learning_clause}"
    return summary


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(99.0, score)), 2)
