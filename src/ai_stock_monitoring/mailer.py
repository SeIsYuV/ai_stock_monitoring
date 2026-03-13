from __future__ import annotations

"""SMTP mail helpers shared by alerts and trade analysis emails."""

from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
import smtplib
from typing import Any


@dataclass(frozen=True)
class EmailDeliveryResult:
    success: bool
    status: str
    error: str | None = None


def send_message(email_settings: Any, subject: str, body: str) -> EmailDeliveryResult:
    """Send one SMTP email using the credentials configured in the Web UI."""
    if not email_settings["recipient_email"]:
        return EmailDeliveryResult(False, "未配置", "未填写收件人邮箱")
    if not email_settings["smtp_server"]:
        return EmailDeliveryResult(False, "未配置", "未填写 SMTP 服务器")
    if not email_settings["sender_email"]:
        return EmailDeliveryResult(False, "未配置", "未填写发件人邮箱")
    if not email_settings["sender_password"]:
        return EmailDeliveryResult(False, "未配置", "未填写授权码或密码")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_settings["sender_email"]
    message["To"] = email_settings["recipient_email"]
    message.set_content(body)

    try:
        with smtplib.SMTP_SSL(email_settings["smtp_server"], 465, timeout=10) as smtp:
            smtp.login(email_settings["sender_email"], email_settings["sender_password"])
            smtp.send_message(message)
    except Exception as exc:  # pragma: no cover - network side effect
        return EmailDeliveryResult(False, "发送失败", str(exc))
    return EmailDeliveryResult(True, "发送成功")


def build_alert_email_body(payload: dict[str, Any]) -> str:
    return (
        f"股票：{payload['symbol']} {payload['display_name']}\n"
        f"触发类型：{payload['trigger_type']}\n"
        f"当前价格：{payload['current_price']:.2f}\n"
        f"指标详情：{payload['detail']}\n"
        f"触发时间：{payload['triggered_at']}\n"
        "\n本系统仅为监控参考，不构成任何投资建议。"
    )


def build_test_email_body(username: str, recipient_email: str) -> str:
    """Render a short SMTP connectivity test email for the current account."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "AI Stock Monitoring 邮箱测试成功。\n\n"
        f"当前账号：{username}\n"
        f"收件邮箱：{recipient_email}\n"
        f"测试时间：{generated_at}\n\n"
        "如果你收到了这封邮件，说明当前 SMTP 配置可正常用于提醒和解封验证码发送。"
    )


def build_trade_analysis_email_body(payload: dict[str, Any]) -> str:
    """Render the latest trade replay result into a readable email body."""

    analysis = payload["analysis"]
    market_snapshot = payload.get("market_snapshot", {})
    portfolio_profile = payload.get("portfolio_profile", {})
    dcf_intrinsic_value = market_snapshot.get("dcf_intrinsic_value")
    dcf_valuation_gap_pct = market_snapshot.get("dcf_valuation_gap_pct")
    dcf_reason = market_snapshot.get("dcf_reason") or "未启用 DCF 模型或当前估值数据不足"
    advice_card = analysis.get("comprehensive_advice_card", {})
    dcf_line = (
        f"内在价值 {dcf_intrinsic_value:.2f}，估值偏差 {dcf_valuation_gap_pct:.2f}%"
        if dcf_intrinsic_value is not None and dcf_valuation_gap_pct is not None
        else dcf_reason
    )

    lines = [
        f"股票：{payload['symbol']}",
        f"分析来源：{payload['provider']} / {payload['model_name']}",
        f"生成时间：{payload['created_at']}",
        "=" * 18,
        f"结论：{analysis['summary']}",
        f"合理性判断：{analysis['judgment']}",
        "",
        "【综合建议卡】",
        f"- 结论｜{advice_card.get('conclusion', analysis['position_advice'])}",
        f"- 买点｜{advice_card.get('buy', analysis.get('recommended_buy_price_range', '暂无明确价位'))}",
        f"- 卖点｜{advice_card.get('sell', analysis.get('recommended_sell_price_range', '暂无明确价位'))}",
        f"- DCF｜{advice_card.get('dcf', dcf_line)}",
        f"- 观望｜{analysis.get('watch_price_range', '暂无明确价位')}",
        "",
        "【判断依据】",
    ]
    lines.extend(f"- {item}" for item in analysis["reasoning"])
    lines.extend(["", "【下一步买点】"])
    lines.extend(f"- {item}" for item in analysis["next_buy_points"])
    lines.extend(["", "【下一步卖点】"])
    lines.extend(f"- {item}" for item in analysis["next_sell_points"])
    lines.extend(["", "【观望关注位】"])
    lines.extend(f"- {item}" for item in analysis.get("watch_points", []))
    if portfolio_profile:
        lines.extend(["", "【组合层面建议】"])
        lines.append(f"- 当前持仓比例：{portfolio_profile.get('holding_ratio', 0)}%")
        lines.append(f"- 模型建议仓位：{portfolio_profile.get('recommended_holding_ratio', '-')}")
        lines.append(f"- 目标仓位中枢：{portfolio_profile.get('target_holding_ratio_mid', '-')}")
        lines.append(f"- 组合建议：{portfolio_profile.get('comprehensive_advice', '-')}")
        lines.extend(f"- {item}" for item in portfolio_profile.get("overall_adjustment_suggestions", []))
        lines.extend(f"- {item}" for item in portfolio_profile.get("priority_reduce_positions", []))
        lines.extend(f"- {item}" for item in portfolio_profile.get("priority_add_positions", []))
    lines.extend(["", "【风控建议】"])
    lines.extend(f"- {item}" for item in analysis["risk_controls"])
    lines.extend(["", analysis["disclaimer"]])
    return "\n".join(lines)


def build_login_unlock_email_body(username: str, verification_code: str, expires_minutes: int) -> str:
    """Render a short email body for login unlock verification."""

    return (
        f"账号：{username}\n"
        f"解封验证码：{verification_code}\n"
        f"有效期：{expires_minutes} 分钟\n\n"
        "如果这不是你本人操作，请尽快修改密码并检查邮箱配置。\n"
        "本系统仅为监控参考，不构成任何投资建议。"
    )



def build_portfolio_review_email_body(payload: dict[str, Any]) -> str:
    """Render an after-close portfolio review email for current holdings."""

    portfolio_profile = payload["portfolio_profile"]
    active_positions = portfolio_profile.get("active_positions", [])
    priority_reduce_positions = portfolio_profile.get("priority_reduce_positions", [])
    priority_add_positions = portfolio_profile.get("priority_add_positions", [])
    overall_adjustment_suggestions = portfolio_profile.get("overall_adjustment_suggestions", [])
    professional_advice = portfolio_profile.get("professional_advice", [])
    risk_reasons = portfolio_profile.get("risk_reasons", [])

    tomorrow_plan: list[str] = []
    if priority_reduce_positions:
        tomorrow_plan.append("明日开盘优先执行减仓计划，先处理偏卖出与高风险仓位。")
    if priority_add_positions:
        tomorrow_plan.append("若盘中回踩关键支撑，可优先对强势标的分批试探加仓。")
    if not tomorrow_plan:
        tomorrow_plan.append("明日以观察为主，先等待关键价位或量价确认后再行动。")

    ranked_positions = sorted(
        active_positions,
        key=lambda item: (
            max(int(item.get("buy_recommendation_level") or 0), int(item.get("sell_recommendation_level") or 0)),
            float(item.get("weight_pct") or 0.0),
        ),
        reverse=True,
    )
    top_priority_positions = ranked_positions[:3]

    red_flag_items: list[str] = []
    for item in active_positions:
        display_name = item.get("display_name", item.get("symbol", "-"))
        if item.get("risk_level") == "高":
            red_flag_items.append(f"{display_name} 风险等级偏高，需优先盯盘。")
        if item.get("action") == "偏卖出" and float(item.get("weight_pct") or 0.0) >= 20:
            red_flag_items.append(f"{display_name} 仓位较重且当前偏卖出，注意止盈/止损执行。")
    red_flag_items.extend(risk_reasons)

    lines = [
        f"账号：{payload['owner_username']}",
        f"交易日：{payload['trade_date']}",
        f"收盘后持仓数：{len(active_positions)}",
        "=" * 18,
        "【组合总览】",
        f"- 组合建议：{portfolio_profile.get('comprehensive_advice', '-')}",
        f"- 当前持仓比例：{portfolio_profile.get('holding_ratio', 0):.2f}%",
        f"- 模型建议仓位：{portfolio_profile.get('recommended_holding_ratio', '-')}",
        f"- 风险等级：{portfolio_profile.get('risk_level', '-')}",
        f"- 持仓风格：{portfolio_profile.get('holding_style', '-')}",
    ]

    lines.extend(["", "【明日计划】"])
    lines.extend(f"- {item}" for item in tomorrow_plan)

    if overall_adjustment_suggestions:
        lines.extend(["", "【总仓位调整】"])
        lines.extend(f"- {item}" for item in overall_adjustment_suggestions)

    if priority_reduce_positions:
        lines.extend(["", "【优先减仓】"])
        lines.extend(f"- {item}" for item in priority_reduce_positions)

    if priority_add_positions:
        lines.extend(["", "【优先加仓】"])
        lines.extend(f"- {item}" for item in priority_add_positions)

    if top_priority_positions:
        lines.extend(["", "【高优先级股票 TOP3】"])
        for item in top_priority_positions:
            lines.append(
                f"- {item.get('display_name', item.get('symbol', '-'))}：动作 {item.get('action', '-')} ｜ 仓位 {float(item.get('weight_pct', 0.0)):.2f}% ｜ 买入 {item.get('buy_recommendation_level', '-')}/10 ｜ 卖出 {item.get('sell_recommendation_level', '-')}/10"
            )

    lines.extend(["", "【持仓逐只建议】"])
    for item in active_positions:
        lines.extend([
            f"- {item.get('display_name', item.get('symbol', '-'))}（{item.get('symbol', '-')}）",
            f"  最新价：{float(item.get('latest_price', 0.0)):.2f} ｜ 仓位：{float(item.get('weight_pct', 0.0)):.2f}% ｜ 动作：{item.get('action', '-')}",
            f"  买点：{item.get('recommended_buy_price_range', '-')} ｜ 买入等级 {item.get('buy_recommendation_level', '-')}/10",
            f"  卖点：{item.get('recommended_sell_price_range', '-')} ｜ 卖出等级 {item.get('sell_recommendation_level', '-')}/10",
            f"  明日关注位：{item.get('watch_price_range', '-')}",
            f"  DCF：{item.get('advice_dcf_line', item.get('dcf_reason', '-'))}",
        ])

    tomorrow_watch_lines = []
    for item in active_positions:
        watch_range = str(item.get('watch_price_range') or '').strip()
        if watch_range and watch_range != '暂无明确价位':
            tomorrow_watch_lines.append(
                f"{item.get('display_name', item.get('symbol', '-'))}：关注 {watch_range}"
            )
    if tomorrow_watch_lines:
        lines.extend(["", "【明日关注位】"])
        lines.extend(f"- {item}" for item in tomorrow_watch_lines)

    if professional_advice:
        lines.extend(["", "【专业综合分析】"])
        lines.extend(f"- {item}" for item in professional_advice)

    if red_flag_items:
        lines.extend(["", "【风险红灯项】"])
        lines.extend(f"- {item}" for item in dict.fromkeys(red_flag_items))

    lines.extend(["", "本系统仅为监控参考，不构成任何投资建议。"])
    return "\n".join(lines)
