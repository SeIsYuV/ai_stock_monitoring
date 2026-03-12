from __future__ import annotations

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
