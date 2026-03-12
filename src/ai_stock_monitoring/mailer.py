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
