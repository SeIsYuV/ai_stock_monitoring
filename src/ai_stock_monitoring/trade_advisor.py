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

        if dividend_yield >= 4.5:
            reasoning.append("股息率较高，若基本面稳定，可作为观察加分项。")

        if int(position_summary["invalid_sell_quantity"]) > 0:
            reasoning.append("存在超出持仓的卖出记录，请先核对交易流水。")
            judgment = "不合理"
            risk_controls.append("卖出数量应小于等于当前实际持仓。")

        if not risk_controls:
            risk_controls.append("控制单笔风险，优先使用预设止损和分批执行。")

        return {
            "summary": "已根据持仓成本、趋势位置和股息率做规则化复盘。",
            "judgment": judgment,
            "reasoning": reasoning,
            "next_buy_points": next_buy_points,
            "next_sell_points": next_sell_points,
            "risk_controls": risk_controls,
            "position_advice": "优先考虑小步试错、分批交易，避免把一次判断当成确定性结论。",
            "confidence": 58,
            "disclaimer": "该分析仅供复盘参考，不构成投资建议。",
        }
