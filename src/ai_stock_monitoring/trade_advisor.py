from __future__ import annotations

"""Trade journal analysis helpers and LLM integration.

这个模块回答两个问题：
1. 用户现在到底持有多少仓位、成本是多少
2. 当前买卖操作放到趋势和均线里看，是否算合理

如果配置了 `OPENAI_API_KEY`，会优先调用大模型；
否则退回到本地规则分析，保证功能始终可用。
"""

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import requests

from .config import AppSettings


BUY_SIGNAL_WEIGHTS: dict[str, int] = {
    "250日线": 2,
    "BOLL中轨": 1,
    "BOLL下轨": 2,
    "股息率": 2,
    "30周/60周均线": 3,
    "量化盈利概率": 4,
}

SELL_SIGNAL_WEIGHTS: dict[str, int] = {
    "BOLL上轨卖出": 2,
    "低股息率卖出": 2,
    "量化走弱卖出": 4,
}

BUY_SIGNAL_SET = set(BUY_SIGNAL_WEIGHTS)
SELL_SIGNAL_SET = set(SELL_SIGNAL_WEIGHTS)


def split_trigger_signals(trigger_state: str | None) -> list[str]:
    if not trigger_state or trigger_state == "正常":
        return []
    return [item for item in str(trigger_state).split("、") if item]


def build_market_action_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Blend buy/sell rules and quant strength into one final action suggestion."""

    trigger_signals = split_trigger_signals(snapshot.get("trigger_state"))
    buy_signals = [item for item in trigger_signals if item in BUY_SIGNAL_SET]
    sell_signals = [item for item in trigger_signals if item in SELL_SIGNAL_SET]

    buy_score = sum(BUY_SIGNAL_WEIGHTS.get(item, 0) for item in buy_signals)
    sell_score = sum(SELL_SIGNAL_WEIGHTS.get(item, 0) for item in sell_signals)

    quant_probability = float(snapshot.get("quant_probability") or 0.0)
    if quant_probability >= 85:
        buy_score += 3
    elif quant_probability >= 70:
        buy_score += 1
    elif quant_probability <= 35:
        sell_score += 4
    elif quant_probability <= 45:
        sell_score += 2

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
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "action": action,
        "action_color": action_color,
        "action_reason": action_reason,
        "dominant_model_label": dominant_model_label,
        "dominant_model_score": round(dominant_model_score, 2),
        "buy_score": buy_score,
        "sell_score": sell_score,
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

    below_or_equal_supports = [level for level in support_levels if latest_price <= 0 or level <= latest_price * 1.03]
    primary_buy_levels = below_or_equal_supports or support_levels[:2]
    primary_sell_levels = [level for level in resistance_levels if level >= latest_price * 0.99] or resistance_levels[-2:]

    breakout_watch = max(_positive_price_levels(ma_250, boll_mid, latest_price), default=0.0)
    defense_watch = min(_positive_price_levels(boll_lower, ma_250, boll_mid), default=0.0)

    buy_range = _format_price_range(primary_buy_levels)
    sell_range = _format_price_range(primary_sell_levels)
    watch_range = _format_price_range(_positive_price_levels(breakout_watch, defense_watch), padding_pct=0.0)

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
    if action == "观望" and breakout_watch > 0 and defense_watch > 0:
        watch_price_plan.append(f"当前更适合观望，重点观察 {watch_range} 这组确认/防守价位。")
    elif action == "偏买入":
        watch_price_plan.append("当前偏买入，但仍应等待回踩确认，不建议直接追涨。")
    elif action == "偏卖出":
        watch_price_plan.append("当前偏卖出，若反弹未能重新站稳关键均线，优先执行减仓计划。")

    return {
        "recommended_buy_price_range": buy_range,
        "recommended_sell_price_range": sell_range,
        "watch_price_range": watch_range,
        "buy_price_plan": buy_price_plan,
        "sell_price_plan": sell_price_plan,
        "watch_price_plan": watch_price_plan,
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
    invalid_sell_quantity = 0

    for trade in trades:
        side = str(trade["side"])
        quantity = int(trade["quantity"])
        price = float(trade["price"])
        amount = price * quantity

        if side == "buy":
            total_buy_amount += amount
            cost_basis_total += amount
            position_quantity += quantity
            continue

        total_sell_amount += amount
        if position_quantity <= 0:
            invalid_sell_quantity += quantity
            continue

        matched_quantity = min(quantity, position_quantity)
        average_cost = cost_basis_total / position_quantity if position_quantity else 0.0
        realized_pnl += (price - average_cost) * matched_quantity
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
                return TradeAnalysisResult(
                    provider="rule-based",
                    model_name="local-fallback",
                    status="fallback",
                    analysis=fallback,
                    error_message=str(exc),
                )

        return TradeAnalysisResult(
            provider="rule-based",
            model_name="local-fallback",
            status="fallback",
            analysis=self._rule_based_analysis(market_snapshot, position_summary),
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
