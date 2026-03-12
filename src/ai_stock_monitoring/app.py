from __future__ import annotations

"""FastAPI web layer for monitoring, journaling and trade analysis."""

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
import uvicorn

from .config import AppSettings, load_settings
from .database import (
    add_monitored_stock,
    add_trade_analysis,
    add_trade_record,
    get_admin_user,
    get_alert_history,
    get_email_settings,
    get_latest_trade_analysis,
    get_snapshots,
    get_unread_alerts,
    initialize_database,
    list_trade_records,
    list_trade_records_for_symbol,
    mark_alert_as_read,
    remove_monitored_stock,
    save_email_settings,
)
from .mailer import send_message
from .monitor import StockMonitor, parse_stock_symbols
from .security import verify_password
from .trade_advisor import TradeAdvisor, build_position_summary


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_database(resolved_settings)
        app.state.monitor_task = asyncio.create_task(_monitor_loop(app))
        try:
            yield
        finally:
            task = app.state.monitor_task
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.monitor = StockMonitor(resolved_settings)
    app.state.trade_advisor = TradeAdvisor(resolved_settings)
    app.state.monitor_task = None

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        if _is_authenticated(request, resolved_settings):
            return RedirectResponse(url="/dashboard", status_code=303)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, error: str | None = None) -> Response:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": error,
                "page_title": "登录",
                "request": request,
            },
        )

    @app.post("/login")
    async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> Response:
        admin_user = get_admin_user(resolved_settings.db_path)
        if username != admin_user["username"] or not verify_password(
            password, admin_user["password_hash"]
        ):
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "error": "用户名或密码错误",
                    "page_title": "登录",
                    "request": request,
                },
                status_code=400,
            )

        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(
            resolved_settings.session_cookie_name,
            "1",
            httponly=True,
            samesite="lax",
        )
        return response

    @app.get("/logout")
    async def logout() -> RedirectResponse:
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(resolved_settings.session_cookie_name)
        return response

    @app.get("/trades", response_class=HTMLResponse)
    async def trades_page(
        request: Request,
        symbol: str | None = None,
        message: str | None = None,
    ) -> Response:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        selected_symbol = (symbol or "").strip()
        trade_records = list_trade_records(resolved_settings.db_path, selected_symbol or None)
        latest_analysis_row = get_latest_trade_analysis(
            resolved_settings.db_path,
            selected_symbol or None,
        )
        latest_analysis = None
        position_summary = None
        if latest_analysis_row is not None:
            latest_analysis = {
                "symbol": latest_analysis_row["symbol"],
                "provider": latest_analysis_row["analysis_provider"],
                "model_name": latest_analysis_row["model_name"],
                "status": latest_analysis_row["status"],
                "error_message": latest_analysis_row["error_message"],
                "created_at": latest_analysis_row["created_at"],
                "analysis": json.loads(latest_analysis_row["analysis_json"]),
            }
            position_summary = json.loads(latest_analysis_row["position_summary"])
        elif selected_symbol:
            position_summary = build_position_summary(
                list_trade_records_for_symbol(resolved_settings.db_path, selected_symbol)
            )

        return templates.TemplateResponse(
            request,
            "trades.html",
            {
                "message": message,
                "page_title": "交易复盘",
                "request": request,
                "selected_symbol": selected_symbol,
                "snapshots": get_snapshots(resolved_settings.db_path),
                "trade_records": trade_records,
                "latest_analysis": latest_analysis,
                "position_summary": position_summary,
            },
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request, message: str | None = None) -> Response:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        monitor: StockMonitor = app.state.monitor
        status = monitor.get_market_status()
        snapshots = get_snapshots(resolved_settings.db_path)
        email_settings = get_email_settings(resolved_settings.db_path)
        unread_alerts = get_unread_alerts(resolved_settings.db_path) if status.is_market_open else []
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "email_settings": email_settings,
                "message": message,
                "monitor": monitor,
                "page_title": "监控面板",
                "request": request,
                "snapshots": snapshots,
                "status": status,
                "system_status": _build_system_status(monitor, email_settings),
                "unread_alerts": unread_alerts,
            },
        )

    @app.post("/stocks")
    async def add_stocks(request: Request, symbols_text: str = Form(...)) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        valid_symbols, invalid_symbols = parse_stock_symbols(symbols_text)
        if not valid_symbols:
            return _redirect_with_message("/dashboard", "请输入6位有效A股代码")
        for symbol in valid_symbols:
            add_monitored_stock(resolved_settings.db_path, symbol)
        await asyncio.to_thread(app.state.monitor.run_cycle)
        if invalid_symbols:
            return _redirect_with_message(
                "/dashboard",
                f"已加入 {len(valid_symbols)} 只股票；以下代码无效：{', '.join(invalid_symbols)}",
            )
        return _redirect_with_message("/dashboard", f"已加入 {len(valid_symbols)} 只股票")

    @app.post("/stocks/{symbol}/delete")
    async def delete_stock(request: Request, symbol: str) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard
        remove_monitored_stock(resolved_settings.db_path, symbol)
        return _redirect_with_message("/dashboard", "股票已删除")

    @app.get("/stocks/{symbol}", response_class=HTMLResponse)
    async def stock_detail(request: Request, symbol: str) -> Response:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard
        monitor: StockMonitor = app.state.monitor
        chart_payload = await asyncio.to_thread(monitor.build_chart_payload, symbol)
        snapshot = next((item for item in get_snapshots(resolved_settings.db_path) if item["symbol"] == symbol), None)
        return templates.TemplateResponse(
            request,
            "stock_detail.html",
            {
                "chart_payload": chart_payload,
                "page_title": f"股票详情 {symbol}",
                "request": request,
                "snapshot": snapshot,
                "symbol": symbol,
            },
        )

    @app.post("/trades")
    async def create_trade_record(
        request: Request,
        symbol: str = Form(...),
        side: str = Form(...),
        price: float = Form(...),
        quantity: int = Form(...),
        traded_at: str = Form(...),
        note: str = Form(""),
    ) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        if side not in {"buy", "sell"}:
            return _redirect_with_message("/trades", "交易方向只能是买入或卖出")
        if not symbol.isdigit() or len(symbol) != 6:
            return _redirect_with_message("/trades", "请输入 6 位股票代码")
        if price <= 0 or quantity <= 0:
            return _redirect_with_message("/trades", "价格和数量必须大于 0")

        add_monitored_stock(resolved_settings.db_path, symbol)
        add_trade_record(
            resolved_settings.db_path,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            traded_at=traded_at,
            note=note.strip(),
        )
        return _redirect_with_message(f"/trades?symbol={symbol}", "交易记录已保存")

    @app.post("/trades/analyze")
    async def analyze_trade_plan(
        request: Request,
        symbol: str = Form(...),
    ) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        if not symbol.isdigit() or len(symbol) != 6:
            return _redirect_with_message("/trades", "请输入 6 位股票代码后再分析")

        trade_rows = list_trade_records_for_symbol(resolved_settings.db_path, symbol)
        if not trade_rows:
            return _redirect_with_message(f"/trades?symbol={symbol}", "请先录入至少一条交易记录")

        monitor: StockMonitor = app.state.monitor
        trade_advisor: TradeAdvisor = app.state.trade_advisor
        try:
            snapshot = await asyncio.to_thread(monitor.build_snapshot, symbol)
            snapshot_payload = {
                "symbol": snapshot.symbol,
                "display_name": snapshot.display_name,
                "latest_price": snapshot.latest_price,
                "ma_250": snapshot.ma_250,
                "ma_30w": snapshot.ma_30w,
                "ma_60w": snapshot.ma_60w,
                "boll_mid": snapshot.boll_mid,
                "dividend_yield": snapshot.dividend_yield,
                "trigger_state": snapshot.trigger_state,
                "trigger_detail": snapshot.trigger_detail,
                "updated_at": snapshot.updated_at.isoformat(),
            }
        except Exception:
            fallback_snapshot = next(
                (item for item in get_snapshots(resolved_settings.db_path) if item["symbol"] == symbol),
                None,
            )
            if fallback_snapshot is None or fallback_snapshot["latest_price"] is None:
                return _redirect_with_message(
                    f"/trades?symbol={symbol}",
                    "当前无法获取行情，请稍后再试",
                )
            snapshot_payload = dict(fallback_snapshot)
        result = await asyncio.to_thread(
            trade_advisor.analyze_symbol,
            symbol,
            snapshot_payload,
            trade_rows,
        )
        add_trade_analysis(
            resolved_settings.db_path,
            symbol=symbol,
            analysis_provider=result.provider,
            model_name=result.model_name,
            position_summary=build_position_summary(trade_rows),
            market_snapshot=snapshot_payload,
            analysis_json=result.analysis,
            status=result.status,
            error_message=result.error_message,
        )
        message = "交易复盘分析已更新"
        if result.error_message:
            message = f"交易复盘分析已更新：{result.error_message}"
        return _redirect_with_message(f"/trades?symbol={symbol}", message)

    @app.post("/settings/email")
    async def update_email_settings(
        request: Request,
        recipient_email: str = Form(""),
        smtp_server: str = Form(""),
        sender_email: str = Form(""),
        sender_password: str = Form(""),
    ) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        save_email_settings(
            db_path=resolved_settings.db_path,
            recipient_email=recipient_email.strip(),
            smtp_server=smtp_server.strip(),
            sender_email=sender_email.strip(),
            sender_password=sender_password.strip(),
        )
        return _redirect_with_message("/dashboard", "邮箱配置已保存")

    @app.post("/settings/email/test")
    async def test_email(request: Request) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard
        email_settings = get_email_settings(resolved_settings.db_path)
        result = send_message(
            email_settings,
            subject="AI Stock Monitoring 测试邮件",
            body="这是一封测试邮件，表示 SMTP 配置可用。",
        )
        notice = "测试邮件发送成功" if result.success else f"测试邮件发送失败：{result.error}"
        return _redirect_with_message("/dashboard", notice)

    @app.post("/alerts/{alert_id}/read")
    async def confirm_alert(request: Request, alert_id: int) -> RedirectResponse:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard
        mark_alert_as_read(resolved_settings.db_path, alert_id)
        return _redirect_with_message("/dashboard", "提醒已确认")

    @app.get("/history", response_class=HTMLResponse)
    async def history_page(
        request: Request,
        symbol: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> Response:
        guard = _require_login(request, resolved_settings)
        if guard is not None:
            return guard

        parsed_from = date.fromisoformat(date_from) if date_from else None
        parsed_to = date.fromisoformat(date_to) if date_to else None
        alerts = get_alert_history(
            resolved_settings.db_path,
            symbol=symbol or None,
            date_from=parsed_from,
            date_to=parsed_to,
        )
        return templates.TemplateResponse(
            request,
            "history.html",
            {
                "alerts": alerts,
                "date_from": date_from or "",
                "date_to": date_to or "",
                "page_title": "历史提醒",
                "request": request,
                "symbol": symbol or "",
            },
        )

    return app


async def _monitor_loop(app: FastAPI) -> None:
    settings: AppSettings = app.state.settings
    monitor: StockMonitor = app.state.monitor
    while True:
        await asyncio.to_thread(monitor.run_cycle)
        await asyncio.sleep(settings.refresh_interval_seconds)


def _build_system_status(monitor: StockMonitor, email_settings: object) -> dict[str, str]:
    mail_status = "已配置" if email_settings["smtp_server"] else "未配置"
    provider_status = "正常" if not monitor.last_error_message else f"异常：{monitor.last_error_message}"
    last_refresh = monitor.last_refresh_at.isoformat(sep=" ", timespec="seconds") if monitor.last_refresh_at else "尚未刷新"
    return {
        "server": "运行中",
        "provider": provider_status,
        "mail": mail_status,
        "last_refresh": last_refresh,
    }


def _is_authenticated(request: Request, settings: AppSettings) -> bool:
    return request.cookies.get(settings.session_cookie_name) == "1"


def _require_login(request: Request, settings: AppSettings) -> RedirectResponse | None:
    if _is_authenticated(request, settings):
        return None
    return RedirectResponse(url="/login", status_code=303)


def _redirect_with_message(path: str, message: str) -> RedirectResponse:
    return RedirectResponse(url=f"{path}?message={quote(message)}", status_code=303)


def run() -> None:
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
