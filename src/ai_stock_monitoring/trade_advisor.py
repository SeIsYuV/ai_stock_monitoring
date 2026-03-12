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
            "dividend_yield": float(snapshot.get("dividend_yield") or 0.0),
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
                        "enum": ["合理", "偏激进", "偏保守", "不合理"],
                    },
                    "reasoning": {"type": "array", "items": {"type": "string"}},
                    "next_buy_points": {"type": "array", "items": {"type": "string"}},
                    "next_sell_points": {"type": "array", "items": {"type": "string"}},
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
                                    "请分析以下 A 股交易记录是否合理，并给出下一步更稳健的买卖点建议。"
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
        position_quantity = int(position_summary["position_quantity"])
        signal_summary = build_market_action_summary(market_snapshot)

        reasoning: list[str] = []
        next_buy_points: list[str] = []
        next_sell_points: list[str] = []
        risk_controls: list[str] = []

        # 规则分析的目标不是代替投顾，而是在没有 LLM 时给出基础复盘建议。
        if position_quantity == 0:
            reasoning.append("当前没有持仓，更适合把重点放在分批建仓和风险预算上。")
        else:
            reasoning.append(
                f"当前持仓 {position_quantity} 股，持仓均价约 {average_cost:.2f}。"
            )

        reasoning.append(signal_summary["action_reason"])

        if ma_250 and latest_price < ma_250:
            reasoning.append("现价低于 250 日线，趋势偏弱，追高并不划算。")
            next_buy_points.append(f"优先等待站回 250 日线附近，即 {ma_250:.2f} 一线再考虑加仓。")
        else:
            reasoning.append("现价位于 250 日线之上，中长期趋势相对更稳。")
            next_buy_points.append("可等待回踩 250 日线或 20 日均线附近，再分批介入。")

        if boll_mid and latest_price < boll_mid:
            next_sell_points.append("若反弹至 BOLL 中轨附近但量能不足，可考虑减仓。")
            risk_controls.append("弱势区间内不要一次性满仓抄底，优先分批建仓。")
        else:
            next_sell_points.append("若后续冲高但无法放量突破前高，可分批止盈。")

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

        if dividend_yield >= 4.5:
            reasoning.append("股息率较高，若基本面稳定，可作为观察加分项。")

        if int(position_summary["invalid_sell_quantity"]) > 0:
            reasoning.append("存在超出持仓的卖出记录，请先核对交易流水。")
            judgment = "不合理"
            risk_controls.append("卖出数量应小于等于当前实际持仓。")

        if signal_summary["action"] == "偏卖出":
            position_advice = "当前卖出信号占优，优先考虑分批止盈或减仓，避免把已有利润重新回吐。"
        elif signal_summary["action"] == "偏买入":
            position_advice = "当前买入信号占优，可考虑等回踩或分批低吸，但不建议一次性重仓追入。"
        else:
            position_advice = "当前买卖信号并存，建议先观望，等待趋势或量化评分进一步拉开差距。"

        if not risk_controls:
            risk_controls.append("控制单笔风险，优先使用预设止损和分批执行。")

        return {
            "summary": f"已结合买入/卖出规则完成综合判断，当前建议：{signal_summary['action']}。",
            "judgment": judgment,
            "reasoning": reasoning,
            "next_buy_points": next_buy_points,
            "next_sell_points": next_sell_points,
            "risk_controls": risk_controls,
            "position_advice": position_advice,
            "confidence": 58,
            "disclaimer": "该分析仅供复盘参考，不构成投资建议。",
        }
