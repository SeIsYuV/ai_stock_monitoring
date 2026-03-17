from __future__ import annotations

"""SMTP mail helpers shared by alerts and trade analysis emails."""

from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
import html
import smtplib
from typing import Any


@dataclass(frozen=True)
class EmailDeliveryResult:
    success: bool
    status: str
    error: str | None = None


def _format_market_context_line(payload: dict[str, Any]) -> str:
    market_environment = str(payload.get("market_environment") or "中性")
    market_bias_score = float(payload.get("market_bias_score") or 0.0)
    industry_name = str(payload.get("industry_name") or "-")
    industry_environment = str(payload.get("industry_environment") or "中性")
    latest_volume_ratio = float(payload.get("latest_volume_ratio") or 1.0)
    earnings_phase = str(payload.get("earnings_phase") or "常规窗口")
    return (
        f"大盘 {market_environment}({market_bias_score:.0f}) ｜ "
        f"行业 {industry_name} {industry_environment} ｜ "
        f"量能比 {latest_volume_ratio:.2f} ｜ 财报节奏 {earnings_phase}"
    )


def send_message(email_settings: Any, subject: str, body: str, html_body: str | None = None) -> EmailDeliveryResult:
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
    if html_body:
        message.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP_SSL(email_settings["smtp_server"], 465, timeout=10) as smtp:
            smtp.login(email_settings["sender_email"], email_settings["sender_password"])
            smtp.send_message(message)
    except Exception as exc:  # pragma: no cover - network side effect
        return EmailDeliveryResult(False, "发送失败", str(exc))
    return EmailDeliveryResult(True, "发送成功")


def _escape_html(value: Any) -> str:
    return html.escape(str(value or ""))


def _render_email_shell(title: str, subtitle: str, sections_html: str, accent: str = "#0f766e") -> str:
    return f"""\
<!DOCTYPE html>
<html lang="zh-CN">
  <body style="margin:0;padding:0;background:#f4f7fb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1f2937;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f7fb;padding:24px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="760" cellspacing="0" cellpadding="0" style="width:760px;max-width:760px;background:#ffffff;border-radius:18px;overflow:hidden;box-shadow:0 12px 40px rgba(15,23,42,0.08);">
            <tr>
              <td style="padding:28px 32px;background:linear-gradient(135deg,{accent} 0%,#0f172a 100%);color:#ffffff;">
                <div style="font-size:13px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.82;">AI Stock Monitoring</div>
                <div style="margin-top:10px;font-size:28px;font-weight:700;line-height:1.25;">{_escape_html(title)}</div>
                <div style="margin-top:8px;font-size:14px;line-height:1.7;opacity:0.92;">{_escape_html(subtitle)}</div>
              </td>
            </tr>
            <tr>
              <td style="padding:28px 32px;">
                {sections_html}
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def _render_email_section(title: str, body_html: str) -> str:
    return (
        "<div style=\"margin:0 0 18px 0;\">"
        f"<div style=\"margin-bottom:10px;font-size:16px;font-weight:700;color:#0f172a;\">{_escape_html(title)}</div>"
        f"<div style=\"border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;background:#ffffff;\">{body_html}</div>"
        "</div>"
    )


def _render_metric_cards(metrics: list[tuple[str, str]]) -> str:
    cards = []
    for label, value in metrics:
        cards.append(
            "<td style=\"padding:0 6px 12px 6px;vertical-align:top;\">"
            "<div style=\"border:1px solid #dbeafe;background:#f8fbff;border-radius:14px;padding:14px 16px;min-height:78px;\">"
            f"<div style=\"font-size:12px;color:#64748b;margin-bottom:6px;\">{_escape_html(label)}</div>"
            f"<div style=\"font-size:18px;font-weight:700;color:#0f172a;line-height:1.45;\">{_escape_html(value)}</div>"
            "</div>"
            "</td>"
        )
    return "<table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\"><tr>{}</tr></table>".format("".join(cards))


def _render_bullet_list(items: list[str], color: str = "#334155") -> str:
    if not items:
        return "<div style=\"font-size:14px;color:#94a3b8;\">暂无</div>"
    rendered = "".join(
        f"<li style=\"margin:0 0 8px 0;line-height:1.7;color:{color};\">{_escape_html(item)}</li>"
        for item in items
    )
    return f"<ul style=\"margin:0;padding-left:20px;\">{rendered}</ul>"


def _render_positions_table(active_positions: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in active_positions:
        action = str(item.get("action") or "-")
        action_color = "#047857" if action == "偏买入" else "#b45309" if action == "观望" else "#b91c1c"
        rows.append(
            "<tr>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\"><div style=\"font-weight:700;color:#0f172a;\">{_escape_html(item.get('display_name', item.get('symbol', '-')))}</div><div style=\"font-size:12px;color:#64748b;\">{_escape_html(item.get('symbol', '-'))}</div></td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;color:{action_color};font-weight:700;\">{_escape_html(action)}</td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\">{float(item.get('weight_pct', 0.0)):.2f}%</td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\">{float(item.get('latest_price', 0.0)):.2f}</td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\">{_escape_html(item.get('recommended_buy_price_range', '-'))}</td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\">{_escape_html(item.get('recommended_sell_price_range', '-'))}</td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\">买 {_escape_html(item.get('buy_recommendation_level', '-'))}/10<br>卖 {_escape_html(item.get('sell_recommendation_level', '-'))}/10</td>"
            f"<td style=\"padding:12px;border-bottom:1px solid #e5e7eb;\">{_escape_html(item.get('watch_price_range', '-'))}</td>"
            "</tr>"
        )
    return (
        "<table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\" style=\"border-collapse:collapse;font-size:13px;\">"
        "<thead><tr>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">股票</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">动作</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">仓位</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">现价</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">买点</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">卖点</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">等级</th>"
        "<th align=\"left\" style=\"padding:12px;background:#eff6ff;border-bottom:1px solid #dbeafe;color:#1e3a8a;\">关注位</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_alert_email_body(payload: dict[str, Any]) -> str:
    lines = [
        f"股票：{payload['symbol']} {payload['display_name']}",
        f"触发类型：{payload['trigger_type']}",
        f"当前价格：{payload['current_price']:.2f}",
    ]
    if any(payload.get(key) is not None for key in ("market_environment", "industry_environment", "latest_volume_ratio", "earnings_phase")):
        lines.append(f"环境因子：{_format_market_context_line(payload)}")
    if payload.get("decision_summary"):
        lines.append(f"总体结论：{payload['decision_summary']}")
    if payload.get("action_reason"):
        lines.append(f"综合原因：{payload['action_reason']}")
    if payload.get("buy_signal_summary"):
        lines.append(str(payload["buy_signal_summary"]))
    if payload.get("sell_signal_summary"):
        lines.append(str(payload["sell_signal_summary"]))
    for item in payload.get("decision_reason_lines", []):
        lines.append(f"观察：{item}")
    if payload.get("trigger_interpretation"):
        lines.append(f"如何理解这条提醒：{payload['trigger_interpretation']}")
    lines.extend([
        f"指标详情：{payload['detail']}",
        f"触发时间：{payload['triggered_at']}",
        "",
        "本系统仅为监控参考，不构成任何投资建议。",
    ])
    return "\n".join(lines)


def build_alert_email_html_body(payload: dict[str, Any]) -> str:
    decision_reason_lines = list(payload.get("decision_reason_lines") or ())
    trigger_interpretation = str(payload.get("trigger_interpretation") or "")
    sections = [
        _render_email_section(
            "提醒摘要",
            _render_metric_cards(
                [
                    ("股票", f"{payload['symbol']} {payload['display_name']}"),
                    ("触发类型", str(payload["trigger_type"])),
                    ("当前价格", f"{float(payload['current_price']):.2f}"),
                    ("触发时间", str(payload["triggered_at"])),
                ]
            ),
        ),
        _render_email_section(
            "本次触发原因",
            "".join(
                [
                    f"<div style=\"font-size:14px;line-height:1.8;color:#334155;\">{_escape_html(payload['detail'])}</div>",
                    (
                        f"<div style=\"margin-top:10px;padding:12px 14px;border-radius:12px;background:#fff7ed;color:#9a3412;font-size:14px;line-height:1.8;\">{_escape_html(trigger_interpretation)}</div>"
                        if trigger_interpretation
                        else ""
                    ),
                ]
            ),
        ),
    ]
    if any(payload.get(key) is not None for key in ("market_environment", "industry_environment", "latest_volume_ratio", "earnings_phase")):
        sections.append(
            _render_email_section(
                "环境与综合判断",
                "".join(
                    [
                        f"<div style=\"font-size:14px;line-height:1.8;color:#334155;\">环境：{_escape_html(_format_market_context_line(payload))}</div>",
                        f"<div style=\"font-size:14px;line-height:1.8;color:#334155;\">总体结论：{_escape_html(payload.get('decision_summary') or payload.get('action_reason') or '-')}</div>",
                        f"<div style=\"font-size:14px;line-height:1.8;color:#334155;\">{_escape_html(payload.get('buy_signal_summary') or '')}</div>",
                        f"<div style=\"font-size:14px;line-height:1.8;color:#334155;\">{_escape_html(payload.get('sell_signal_summary') or '')}</div>",
                        _render_bullet_list(decision_reason_lines) if decision_reason_lines else "",
                    ]
                ),
            )
        )
    return _render_email_shell(
        title=f"{payload['symbol']} {payload['display_name']} 提醒",
        subtitle=f"本次触发类型：{payload['trigger_type']}",
        sections_html="".join(sections),
        accent="#1d4ed8" if "卖出" not in str(payload.get("trigger_type") or "") else "#b91c1c",
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
        f"- 环境｜{_format_market_context_line(market_snapshot)}",
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


def build_trade_analysis_email_html_body(payload: dict[str, Any]) -> str:
    analysis = payload["analysis"]
    market_snapshot = payload.get("market_snapshot", {})
    portfolio_profile = payload.get("portfolio_profile", {})
    advice_card = analysis.get("comprehensive_advice_card", {})
    summary_cards = _render_metric_cards(
        [
            ("股票", str(payload["symbol"])),
            ("合理性判断", str(analysis["judgment"])),
            ("买入等级", f"{analysis.get('buy_recommendation_level', '-')}/10"),
            ("卖出等级", f"{analysis.get('sell_recommendation_level', '-')}/10"),
        ]
    )
    sections = [
        _render_email_section("结论摘要", summary_cards),
        _render_email_section(
            "综合建议卡",
            "".join(
                [
                    f"<div style=\"margin-bottom:10px;font-size:14px;color:#334155;\"><strong>结论：</strong>{_escape_html(advice_card.get('conclusion', analysis['position_advice']))}</div>",
                    f"<div style=\"margin-bottom:10px;font-size:14px;color:#334155;\"><strong>买点：</strong>{_escape_html(advice_card.get('buy', analysis.get('recommended_buy_price_range', '-')))}</div>",
                    f"<div style=\"margin-bottom:10px;font-size:14px;color:#334155;\"><strong>卖点：</strong>{_escape_html(advice_card.get('sell', analysis.get('recommended_sell_price_range', '-')))}</div>",
                    f"<div style=\"margin-bottom:10px;font-size:14px;color:#334155;\"><strong>DCF：</strong>{_escape_html(advice_card.get('dcf', market_snapshot.get('dcf_reason', '-')))}</div>",
                    f"<div style=\"margin-bottom:10px;font-size:14px;color:#334155;\"><strong>环境：</strong>{_escape_html(_format_market_context_line(market_snapshot))}</div>",
                    f"<div style=\"font-size:14px;color:#334155;\"><strong>观望位：</strong>{_escape_html(analysis.get('watch_price_range', '-'))}</div>",
                ]
            ),
        ),
        _render_email_section("判断依据", _render_bullet_list(list(analysis.get("reasoning") or []))),
        _render_email_section("下一步买点", _render_bullet_list(list(analysis.get("next_buy_points") or []), color="#065f46")),
        _render_email_section("下一步卖点", _render_bullet_list(list(analysis.get("next_sell_points") or []), color="#991b1b")),
        _render_email_section("观望关注位", _render_bullet_list(list(analysis.get("watch_points") or []))),
    ]
    if portfolio_profile:
        sections.append(
            _render_email_section(
                "组合层面建议",
                "".join(
                    [
                        _render_metric_cards(
                            [
                                ("当前持仓比例", f"{portfolio_profile.get('holding_ratio', 0)}%"),
                                ("建议仓位", str(portfolio_profile.get('recommended_holding_ratio', '-'))),
                                ("目标仓位中枢", str(portfolio_profile.get('target_holding_ratio_mid', '-'))),
                                ("组合建议", str(portfolio_profile.get('comprehensive_advice', '-'))),
                            ]
                        ),
                        _render_bullet_list(
                            list(portfolio_profile.get("overall_adjustment_suggestions") or [])
                            + list(portfolio_profile.get("priority_reduce_positions") or [])
                            + list(portfolio_profile.get("priority_add_positions") or [])
                        ),
                    ]
                ),
            )
        )
    sections.append(_render_email_section("风控建议", _render_bullet_list(list(analysis.get("risk_controls") or []), color="#7f1d1d")))
    sections.append(f"<div style=\"margin-top:10px;font-size:12px;color:#94a3b8;line-height:1.7;\">{_escape_html(analysis['disclaimer'])}</div>")
    return _render_email_shell(
        title=f"{payload['symbol']} 交易复盘",
        subtitle=f"{payload['provider']} / {payload['model_name']} ｜ {payload['created_at']}",
        sections_html="".join(sections),
        accent="#0f766e",
    )


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
            f"  结论：{item.get('decision_summary', item.get('comprehensive_advice', '-'))}",
            f"  原因：{item.get('action_reason', '-')}",
            f"  {item.get('buy_signal_summary', '买入侧信号：暂无显著信号')}",
            f"  {item.get('sell_signal_summary', '卖出侧信号：暂无显著信号')}",
            f"  环境：{_format_market_context_line(item)}",
            f"  DCF：{item.get('advice_dcf_line', item.get('dcf_reason', '-'))}",
        ])
        for reason in item.get("decision_reason_lines", []):
            lines.append(f"  观察：{reason}")

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


def build_portfolio_review_email_html_body(payload: dict[str, Any]) -> str:
    portfolio_profile = payload["portfolio_profile"]
    active_positions = list(portfolio_profile.get("active_positions") or [])
    sections = [
        _render_email_section(
            "组合总览",
            _render_metric_cards(
                [
                    ("组合建议", str(portfolio_profile.get("comprehensive_advice", "-"))),
                    ("当前持仓比例", f"{float(portfolio_profile.get('holding_ratio', 0.0)):.2f}%"),
                    ("建议仓位", str(portfolio_profile.get("recommended_holding_ratio", "-"))),
                    ("风险等级", str(portfolio_profile.get("risk_level", "-"))),
                ]
            ),
        ),
        _render_email_section("明日计划", _render_bullet_list(list(filter(None, [
            "明日开盘优先执行减仓计划，先处理偏卖出与高风险仓位。" if portfolio_profile.get("priority_reduce_positions") else "",
            "若盘中回踩关键支撑，可优先对强势标的分批试探加仓。" if portfolio_profile.get("priority_add_positions") else "",
            "" if (portfolio_profile.get("priority_reduce_positions") or portfolio_profile.get("priority_add_positions")) else "明日以观察为主，先等待关键价位或量价确认后再行动。",
        ])))),
    ]
    if portfolio_profile.get("overall_adjustment_suggestions"):
        sections.append(_render_email_section("总仓位调整", _render_bullet_list(list(portfolio_profile.get("overall_adjustment_suggestions") or []))))
    if portfolio_profile.get("priority_reduce_positions"):
        sections.append(_render_email_section("优先减仓", _render_bullet_list(list(portfolio_profile.get("priority_reduce_positions") or []), color="#991b1b")))
    if portfolio_profile.get("priority_add_positions"):
        sections.append(_render_email_section("优先加仓", _render_bullet_list(list(portfolio_profile.get("priority_add_positions") or []), color="#065f46")))
    if active_positions:
        sections.append(_render_email_section("持仓对比表", _render_positions_table(active_positions)))
        detail_blocks = []
        for item in active_positions:
            detail_blocks.append(
                "<div style=\"margin-bottom:14px;padding:14px 16px;border:1px solid #e5e7eb;border-radius:12px;background:#fafcff;\">"
                f"<div style=\"font-size:16px;font-weight:700;color:#0f172a;margin-bottom:8px;\">{_escape_html(item.get('display_name', item.get('symbol', '-')))}（{_escape_html(item.get('symbol', '-'))}）</div>"
                f"<div style=\"font-size:14px;color:#334155;line-height:1.8;\"><strong>结论：</strong>{_escape_html(item.get('decision_summary', '-'))}</div>"
                f"<div style=\"font-size:14px;color:#334155;line-height:1.8;\"><strong>原因：</strong>{_escape_html(item.get('action_reason', '-'))}</div>"
                f"<div style=\"font-size:14px;color:#334155;line-height:1.8;\">{_escape_html(item.get('buy_signal_summary', '买入侧信号：暂无显著信号'))}</div>"
                f"<div style=\"font-size:14px;color:#334155;line-height:1.8;\">{_escape_html(item.get('sell_signal_summary', '卖出侧信号：暂无显著信号'))}</div>"
                f"{_render_bullet_list(list(item.get('decision_reason_lines') or []))}"
                f"<div style=\"margin-top:10px;font-size:13px;color:#475569;line-height:1.8;\">环境：{_escape_html(_format_market_context_line(item))}</div>"
                f"<div style=\"font-size:13px;color:#475569;line-height:1.8;\">DCF：{_escape_html(item.get('advice_dcf_line', item.get('dcf_reason', '-')))}</div>"
                "</div>"
            )
        sections.append(_render_email_section("逐只拆解", "".join(detail_blocks)))
    if portfolio_profile.get("professional_advice"):
        sections.append(_render_email_section("专业综合分析", _render_bullet_list(list(portfolio_profile.get("professional_advice") or []))))
    if portfolio_profile.get("risk_reasons"):
        sections.append(_render_email_section("风险红灯项", _render_bullet_list(list(portfolio_profile.get("risk_reasons") or []), color="#991b1b")))
    sections.append("<div style=\"margin-top:10px;font-size:12px;color:#94a3b8;line-height:1.7;\">本系统仅为监控参考，不构成任何投资建议。</div>")
    return _render_email_shell(
        title=f"{payload['owner_username']} 收盘持仓复盘",
        subtitle=f"交易日：{payload['trade_date']} ｜ 收盘后持仓数：{len(active_positions)}",
        sections_html="".join(sections),
        accent="#1d4ed8",
    )
