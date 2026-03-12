from __future__ import annotations

"""SMTP mail helpers shared by alerts and trade analysis emails."""

from dataclasses import dataclass
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


def build_trade_analysis_email_body(payload: dict[str, Any]) -> str:
    """Render the latest trade replay result into a readable email body."""

    analysis = payload["analysis"]
    lines = [
        f"股票：{payload['symbol']}",
        f"分析来源：{payload['provider']} / {payload['model_name']}",
        f"生成时间：{payload['created_at']}",
        f"结论：{analysis['summary']}",
        f"合理性判断：{analysis['judgment']}",
        f"仓位建议：{analysis['position_advice']}",
        "",
        "判断依据：",
    ]
    lines.extend(f"- {item}" for item in analysis["reasoning"])
    lines.extend(["", "下一步买点："])
    lines.extend(f"- {item}" for item in analysis["next_buy_points"])
    lines.extend(["", "下一步卖点："])
    lines.extend(f"- {item}" for item in analysis["next_sell_points"])
    lines.extend(["", "风控建议："])
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
