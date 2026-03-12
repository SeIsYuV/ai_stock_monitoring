from __future__ import annotations

"""FastAPI web layer for monitoring, journaling and trade analysis.

新人接手建议先从本文件读起：
1. `create_app` 负责组装整个 Web 应用
2. `lifespan` 负责启动时初始化数据库和后台轮询
3. `/dashboard`、`/trades`、`/history` 分别对应三个核心页面
4. 多账号逻辑通过 `_get_authenticated_user` 和数据库里的 `owner_username` 实现数据隔离
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
import json
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from openpyxl import Workbook
import uvicorn

from .config import AppSettings, load_settings
from .database import (
    add_monitored_stock,
    add_trade_analysis,
    add_trade_record,
    clear_login_guard_state,
    clear_login_guard_states_for_username,
    consume_login_unlock_code,
    create_user,
    get_admin_user,
    get_alert_history,
    get_email_settings,
    get_latest_trade_analysis,
    get_login_guard_state,
    get_login_unlock_code,
    get_quant_settings,
    get_snapshot,
    get_snapshots,
    get_unread_alerts,
    get_user,
    initialize_database,
    list_trade_records,
    list_trade_records_for_symbol,
    list_users,
    mark_alert_as_read,
    record_failed_login,
    remove_monitored_stock,
    save_email_settings,
    save_login_unlock_code,
    save_quant_settings,
    update_user_password,
)
from .mailer import build_login_unlock_email_body, build_trade_analysis_email_body, send_message
from .monitor import StockMonitor, parse_stock_symbols
from .quant import available_quant_models, normalize_selected_models
from .security import hash_password, password_hash_needs_rehash, verify_password
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
        if _get_authenticated_user(request, resolved_settings) is not None:
            return RedirectResponse(url="/dashboard", status_code=303)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(
        request: Request,
        error: str | None = None,
        message: str | None = None,
        unlock_username: str | None = None,
    ) -> Response:
        return _render_login_page(
            request,
            error=error,
            message=message,
            unlock_username=unlock_username or "",
        )

    @app.post("/login")
    async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> Response:
        normalized_username = username.strip()
        client_subject = _login_subject(request, normalized_username)
        guard_state = get_login_guard_state(resolved_settings.db_path, client_subject)
        if guard_state and guard_state["locked_until"]:
            locked_until = datetime.fromisoformat(guard_state["locked_until"])
            now = datetime.now(locked_until.tzinfo)
            if locked_until > now:
                remaining_seconds = max(1, int((locked_until - now).total_seconds()))
                remaining_minutes = (remaining_seconds + 59) // 60
                return _render_login_page(
                    request,
                    error=(
                        f"登录失败次数过多，当前访问已被临时锁定，请约 {remaining_minutes} 分钟后再试。"
                        " 如已绑定邮箱，可直接在下方发送验证码解封。"
                    ),
                    unlock_username=normalized_username,
                    status_code=429,
                )

        user = get_user(resolved_settings.db_path, normalized_username)
        if user is None or not verify_password(password, user["password_hash"]):
            record_failed_login(resolved_settings.db_path, client_subject)
            return _render_login_page(
                request,
                error="用户名或密码错误",
                unlock_username=normalized_username,
                status_code=400,
            )

        clear_login_guard_state(resolved_settings.db_path, client_subject)
        if password_hash_needs_rehash(user["password_hash"]):
            update_user_password(resolved_settings.db_path, user["username"], hash_password(password))
            user = get_user(resolved_settings.db_path, normalized_username)
        if user is None:
            return _render_login_page(
                request,
                error="登录态初始化失败，请重试",
                unlock_username=normalized_username,
                status_code=500,
            )

        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(
            resolved_settings.session_cookie_name,
            _build_session_cookie_value(user),
            httponly=True,
            samesite="lax",
        )
        return response

    @app.post("/login/unlock/request")
    async def request_login_unlock(
        request: Request,
        username: str = Form(...),
    ) -> Response:
        normalized_username = username.strip()
        user = get_user(resolved_settings.db_path, normalized_username)
        if user is None:
            return _render_login_page(
                request,
                error="未找到该账号，无法发送解封验证码。",
                unlock_username=normalized_username,
                status_code=404,
            )

        email_settings = get_email_settings(resolved_settings.db_path, normalized_username)
        target_email = _resolve_unlock_email_target(email_settings)
        if not target_email:
            return _render_login_page(
                request,
                error="该账号尚未配置可用邮箱，暂时无法自助解封，请联系管理员。",
                unlock_username=normalized_username,
                status_code=400,
            )

        verification_code = f"{secrets.randbelow(1_000_000):06d}"
        expires_minutes = 10
        expires_at = (datetime.now(UTC) + timedelta(minutes=expires_minutes)).isoformat()
        save_login_unlock_code(
            resolved_settings.db_path,
            normalized_username,
            verification_code,
            expires_at,
        )
        unlock_email_settings = dict(email_settings)
        unlock_email_settings["recipient_email"] = target_email
        result = send_message(
            unlock_email_settings,
            subject=f"[登录解封] {normalized_username} 验证码",
            body=build_login_unlock_email_body(normalized_username, verification_code, expires_minutes),
        )
        if not result.success:
            return _render_login_page(
                request,
                error=f"解封邮件发送失败：{result.error}",
                unlock_username=normalized_username,
                status_code=400,
            )
        masked_email = _mask_email_address(target_email)
        return _render_login_page(
            request,
            message=f"验证码已发送到 {masked_email}，请输入验证码完成解封。",
            unlock_username=normalized_username,
        )

    @app.post("/login/unlock/confirm")
    async def confirm_login_unlock(
        request: Request,
        username: str = Form(...),
        verification_code: str = Form(...),
    ) -> Response:
        normalized_username = username.strip()
        unlock_row = get_login_unlock_code(resolved_settings.db_path, normalized_username)
        if unlock_row is None:
            return _render_login_page(
                request,
                error="请先发送解封验证码。",
                unlock_username=normalized_username,
                status_code=400,
            )
        if unlock_row["consumed_at"]:
            return _render_login_page(
                request,
                error="该验证码已使用，请重新发送新的验证码。",
                unlock_username=normalized_username,
                status_code=400,
            )
        expires_at = datetime.fromisoformat(unlock_row["expires_at"])
        if expires_at <= datetime.now(expires_at.tzinfo):
            return _render_login_page(
                request,
                error="验证码已过期，请重新发送新的验证码。",
                unlock_username=normalized_username,
                status_code=400,
            )
        if verification_code.strip() != str(unlock_row["verification_code"]):
            return _render_login_page(
                request,
                error="验证码不正确，请重新输入。",
                unlock_username=normalized_username,
                status_code=400,
            )

        clear_login_guard_states_for_username(resolved_settings.db_path, normalized_username)
        consume_login_unlock_code(resolved_settings.db_path, normalized_username)
        return _render_login_page(
            request,
            message="邮箱验证成功，账号锁定已解除，请重新登录。",
            unlock_username=normalized_username,
        )

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
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        selected_symbol = (symbol or "").strip()
        owner_username = current_user["username"]
        trade_records = list_trade_records(resolved_settings.db_path, owner_username, selected_symbol or None)
        latest_analysis_row = get_latest_trade_analysis(
            resolved_settings.db_path,
            owner_username,
            selected_symbol or None,
        )
        latest_analysis = None
        position_summary = None
        if latest_analysis_row is not None:
            latest_analysis = _serialize_latest_analysis_row(latest_analysis_row)
            position_summary = json.loads(latest_analysis_row["position_summary"])
        elif selected_symbol:
            position_summary = build_position_summary(
                list_trade_records_for_symbol(resolved_settings.db_path, owner_username, selected_symbol)
            )

        return templates.TemplateResponse(
            request,
            "trades.html",
            _base_template_context(
                request,
                current_user,
                message=message,
                page_title="交易复盘",
                selected_symbol=selected_symbol,
                snapshots=get_snapshots(resolved_settings.db_path, owner_username),
                trade_records=trade_records,
                latest_analysis=latest_analysis,
                position_summary=position_summary,
            ),
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/trades/export")
    async def export_trades(request: Request, symbol: str | None = None) -> Response:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        owner_username = current_user["username"]
        selected_symbol = (symbol or "").strip() or None
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "交易流水"
        worksheet.append(["成交时间", "股票代码", "方向", "价格", "数量", "备注", "录入时间"])
        trade_records = list_trade_records(resolved_settings.db_path, owner_username, selected_symbol)
        for trade in trade_records:
            worksheet.append(
                [
                    trade["traded_at"],
                    trade["symbol"],
                    "买入" if trade["side"] == "buy" else "卖出",
                    float(trade["price"]),
                    int(trade["quantity"]),
                    trade["note"] or "",
                    trade["created_at"],
                ]
            )

        latest_analysis_row = get_latest_trade_analysis(resolved_settings.db_path, owner_username, selected_symbol)
        if latest_analysis_row is not None:
            analysis_sheet = workbook.create_sheet(title="最新分析")
            analysis = json.loads(latest_analysis_row["analysis_json"])
            analysis_sheet.append(["股票代码", latest_analysis_row["symbol"]])
            analysis_sheet.append(["分析来源", latest_analysis_row["analysis_provider"]])
            analysis_sheet.append(["模型", latest_analysis_row["model_name"]])
            analysis_sheet.append(["状态", latest_analysis_row["status"]])
            analysis_sheet.append(["生成时间", latest_analysis_row["created_at"]])
            analysis_sheet.append(["结论", analysis["summary"]])
            analysis_sheet.append(["合理性判断", analysis["judgment"]])
            analysis_sheet.append(["仓位建议", analysis["position_advice"]])
            analysis_sheet.append(["置信度", analysis["confidence"]])

        buffer = BytesIO()
        workbook.save(buffer)
        buffer.seek(0)
        filename_symbol = selected_symbol or "all"
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="trade_records_{owner_username}_{filename_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx"'
                )
            },
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request, message: str | None = None) -> Response:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        monitor: StockMonitor = app.state.monitor
        owner_username = current_user["username"]
        status = monitor.get_market_status()
        snapshots = await _load_dashboard_snapshots(app, owner_username)
        email_settings = get_email_settings(resolved_settings.db_path, owner_username)
        quant_settings = get_quant_settings(resolved_settings.db_path, owner_username)
        unread_alerts = get_unread_alerts(resolved_settings.db_path, owner_username) if status.is_market_open else []
        selected_models = normalize_selected_models(json.loads(quant_settings["selected_models"] or "[]"))
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            _base_template_context(
                request,
                current_user,
                message=message,
                page_title="监控面板",
                admin_username=get_admin_user(resolved_settings.db_path)["username"],
                email_settings=email_settings,
                quant_settings=quant_settings,
                quant_model_options=available_quant_models(),
                quant_selected_models=selected_models,
                is_admin=bool(current_user["is_admin"]),
                monitor=monitor,
                price_column_label="最新价" if status.is_market_open else "最近收盘/最新可用价",
                snapshots=snapshots,
                status=status,
                system_status=_build_system_status(monitor, email_settings),
                unread_alerts=unread_alerts,
                users=list_users(resolved_settings.db_path) if current_user["is_admin"] else [],
            ),
        )

    @app.post("/stocks")
    async def add_stocks(request: Request, symbols_text: str = Form(...)) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        owner_username = current_user["username"]
        valid_symbols, invalid_symbols = parse_stock_symbols(symbols_text)
        if not valid_symbols:
            return _redirect_with_message("/dashboard", "请输入6位有效A股代码")
        refresh_failed_symbols: list[str] = []
        for symbol in valid_symbols:
            add_monitored_stock(resolved_settings.db_path, owner_username, symbol)
            try:
                await asyncio.to_thread(app.state.monitor.refresh_symbol_snapshot, owner_username, symbol)
            except Exception:
                refresh_failed_symbols.append(symbol)
        if invalid_symbols:
            return _redirect_with_message(
                "/dashboard",
                f"已加入 {len(valid_symbols)} 只股票；以下代码无效：{', '.join(invalid_symbols)}",
            )
        if refresh_failed_symbols:
            return _redirect_with_message(
                "/dashboard",
                f"已加入 {len(valid_symbols)} 只股票；以下股票暂未拉到最新数据：{', '.join(refresh_failed_symbols)}",
            )
        return _redirect_with_message("/dashboard", f"已加入 {len(valid_symbols)} 只股票")

    @app.post("/stocks/{symbol}/delete")
    async def delete_stock(request: Request, symbol: str) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user
        remove_monitored_stock(resolved_settings.db_path, current_user["username"], symbol)
        return _redirect_with_message("/dashboard", "股票已删除")

    @app.get("/stocks/{symbol}", response_class=HTMLResponse)
    async def stock_detail(request: Request, symbol: str) -> Response:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user
        owner_username = current_user["username"]
        snapshot = get_snapshot(resolved_settings.db_path, owner_username, symbol)
        if snapshot is None:
            return _redirect_with_message("/dashboard", "当前账号下未找到该股票")
        monitor: StockMonitor = app.state.monitor
        chart_payload = await asyncio.to_thread(monitor.build_chart_payload, symbol)
        return templates.TemplateResponse(
            request,
            "stock_detail.html",
            _base_template_context(
                request,
                current_user,
                chart_payload=chart_payload,
                page_title=f"股票详情 {symbol}",
                snapshot=snapshot,
                symbol=symbol,
            ),
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
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        owner_username = current_user["username"]
        if side not in {"buy", "sell"}:
            return _redirect_with_message("/trades", "交易方向只能是买入或卖出")
        if not symbol.isdigit() or len(symbol) != 6:
            return _redirect_with_message("/trades", "请输入 6 位股票代码")
        if price <= 0 or quantity <= 0:
            return _redirect_with_message("/trades", "价格和数量必须大于 0")

        add_monitored_stock(resolved_settings.db_path, owner_username, symbol)
        add_trade_record(
            resolved_settings.db_path,
            owner_username=owner_username,
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
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        owner_username = current_user["username"]
        if not symbol.isdigit() or len(symbol) != 6:
            return _redirect_with_message("/trades", "请输入 6 位股票代码后再分析")

        trade_rows = list_trade_records_for_symbol(resolved_settings.db_path, owner_username, symbol)
        if not trade_rows:
            return _redirect_with_message(f"/trades?symbol={symbol}", "请先录入至少一条交易记录")

        monitor: StockMonitor = app.state.monitor
        trade_advisor: TradeAdvisor = app.state.trade_advisor

        try:
            snapshot = await asyncio.to_thread(monitor.refresh_symbol_snapshot, owner_username, symbol)
            snapshot_payload = {
                "symbol": snapshot.symbol,
                "display_name": snapshot.display_name,
                "latest_price": snapshot.latest_price,
                "ma_250": snapshot.ma_250,
                "ma_30w": snapshot.ma_30w,
                "ma_60w": snapshot.ma_60w,
                "boll_mid": snapshot.boll_mid,
                "dividend_yield": snapshot.dividend_yield,
                "quant_probability": snapshot.quant_probability,
                "trigger_state": snapshot.trigger_state,
                "trigger_detail": snapshot.trigger_detail,
                "updated_at": snapshot.updated_at.isoformat(),
            }
        except Exception:
            fallback_snapshot = get_snapshot(resolved_settings.db_path, owner_username, symbol)
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
            owner_username=owner_username,
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
        email_result = _send_trade_analysis_email_if_configured(
            db_path=resolved_settings.db_path,
            owner_username=owner_username,
            symbol=symbol,
        )
        if email_result is not None:
            message = f"{message}；{email_result}"
        if result.error_message:
            message = f"交易复盘分析已更新：{result.error_message}"
        return _redirect_with_message(f"/trades?symbol={symbol}", message)

    @app.post("/trades/email-analysis")
    async def email_trade_analysis(
        request: Request,
        symbol: str = Form(...),
    ) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        if not symbol.isdigit() or len(symbol) != 6:
            return _redirect_with_message("/trades", "请输入 6 位股票代码后再发送")

        result_message = _send_trade_analysis_email_if_configured(
            resolved_settings.db_path,
            current_user["username"],
            symbol,
        )
        if result_message is None:
            result_message = "请先生成至少一条复盘分析后再发送"
        return _redirect_with_message(f"/trades?symbol={symbol}", result_message)

    @app.post("/settings/email")
    async def update_email_settings(
        request: Request,
        recipient_email: str = Form(""),
        smtp_server: str = Form(""),
        sender_email: str = Form(""),
        sender_password: str = Form(""),
    ) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        save_email_settings(
            db_path=resolved_settings.db_path,
            owner_username=current_user["username"],
            recipient_email=recipient_email.strip(),
            smtp_server=smtp_server.strip(),
            sender_email=sender_email.strip(),
            sender_password=sender_password.strip(),
        )
        return _redirect_with_message("/dashboard", "邮箱配置已保存")

    @app.post("/settings/quant")
    async def update_quant_settings(
        request: Request,
        enabled: str | None = Form(None),
        probability_threshold: float = Form(90),
        selected_models: list[str] | None = Form(None),
    ) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        normalized_models = list(normalize_selected_models(selected_models or []))
        threshold = max(50.0, min(99.0, float(probability_threshold)))
        save_quant_settings(
            db_path=resolved_settings.db_path,
            owner_username=current_user["username"],
            enabled=enabled is not None,
            probability_threshold=threshold,
            selected_models=normalized_models,
        )
        snapshots = get_snapshots(resolved_settings.db_path, current_user["username"])
        for item in snapshots:
            try:
                await asyncio.to_thread(app.state.monitor.refresh_symbol_snapshot, current_user["username"], item["symbol"])
            except Exception:
                continue
        return _redirect_with_message("/dashboard", "量化提醒设置已保存")

    @app.post("/settings/admin-password")
    async def update_account_password_route(
        request: Request,
        current_password: str = Form(...),
        new_password: str = Form(...),
        confirm_password: str = Form(...),
    ) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        if not verify_password(current_password, current_user["password_hash"]):
            return _redirect_with_message("/dashboard", "当前密码不正确")
        if len(new_password) < 8:
            return _redirect_with_message("/dashboard", "新密码至少需要 8 位")
        if new_password != confirm_password:
            return _redirect_with_message("/dashboard", "两次输入的新密码不一致")
        if new_password == current_password:
            return _redirect_with_message("/dashboard", "新密码不能和当前密码相同")

        update_user_password(resolved_settings.db_path, current_user["username"], hash_password(new_password))
        refreshed_user = get_user(resolved_settings.db_path, current_user["username"])
        response = _redirect_with_message("/dashboard", "当前账号密码已更新，请牢记新密码")
        if refreshed_user is not None:
            response.set_cookie(
                resolved_settings.session_cookie_name,
                _build_session_cookie_value(refreshed_user),
                httponly=True,
                samesite="lax",
            )
        return response

    @app.post("/settings/users")
    async def create_user_route(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        confirm_password: str = Form(...),
        is_admin: str | None = Form(None),
    ) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user
        if not current_user["is_admin"]:
            return _redirect_with_message("/dashboard", "只有管理员可以新增账号")

        normalized_username = username.strip()
        if len(normalized_username) < 3 or len(normalized_username) > 32:
            return _redirect_with_message("/dashboard", "账号长度需在 3 到 32 位之间")
        if not all(char.isalnum() or char in {"_", "-", "."} for char in normalized_username):
            return _redirect_with_message("/dashboard", "账号仅支持字母、数字、下划线、中划线和点")
        if len(password) < 8:
            return _redirect_with_message("/dashboard", "新账号密码至少需要 8 位")
        if password != confirm_password:
            return _redirect_with_message("/dashboard", "两次输入的账号密码不一致")
        if get_user(resolved_settings.db_path, normalized_username) is not None:
            return _redirect_with_message("/dashboard", "该账号已存在")
        try:
            create_user(
                resolved_settings.db_path,
                username=normalized_username,
                password_hash=hash_password(password),
                is_admin=is_admin is not None,
            )
        except sqlite3.IntegrityError:
            return _redirect_with_message("/dashboard", "该账号已存在")
        return _redirect_with_message("/dashboard", f"账号 {normalized_username} 已创建")

    @app.post("/settings/email/test")
    async def test_email(request: Request) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user
        email_settings = get_email_settings(resolved_settings.db_path, current_user["username"])
        result = send_message(
            email_settings,
            subject="AI Stock Monitoring 测试邮件",
            body="这是一封测试邮件，表示当前账号的 SMTP 配置可用。",
        )
        notice = "测试邮件发送成功" if result.success else f"测试邮件发送失败：{result.error}"
        return _redirect_with_message("/dashboard", notice)

    @app.post("/alerts/{alert_id}/read")
    async def confirm_alert(request: Request, alert_id: int) -> RedirectResponse:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user
        mark_alert_as_read(resolved_settings.db_path, current_user["username"], alert_id)
        return _redirect_with_message("/dashboard", "提醒已确认")

    @app.get("/history", response_class=HTMLResponse)
    async def history_page(
        request: Request,
        symbol: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> Response:
        current_user = _require_login(request, resolved_settings)
        if isinstance(current_user, RedirectResponse):
            return current_user

        parsed_from = date.fromisoformat(date_from) if date_from else None
        parsed_to = date.fromisoformat(date_to) if date_to else None
        alerts = get_alert_history(
            resolved_settings.db_path,
            owner_username=current_user["username"],
            symbol=symbol or None,
            date_from=parsed_from,
            date_to=parsed_to,
        )
        return templates.TemplateResponse(
            request,
            "history.html",
            _base_template_context(
                request,
                current_user,
                alerts=alerts,
                date_from=date_from or "",
                date_to=date_to or "",
                page_title="历史提醒",
                symbol=symbol or "",
            ),
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


def _base_template_context(request: Request, current_user: object, **kwargs: object) -> dict[str, object]:
    context = {
        "request": request,
        "current_user": current_user,
        "is_admin": bool(current_user["is_admin"]),
    }
    context.update(kwargs)
    return context


def _render_login_page(
    request: Request,
    error: str | None = None,
    message: str | None = None,
    unlock_username: str = "",
    status_code: int = 200,
) -> Response:
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": error,
            "message": message,
            "page_title": "登录",
            "request": request,
            "unlock_username": unlock_username,
        },
        status_code=status_code,
    )


def _resolve_unlock_email_target(email_settings: object) -> str:
    return str(email_settings["recipient_email"] or email_settings["sender_email"] or "").strip()


def _mask_email_address(email: str) -> str:
    if "@" not in email:
        return email
    local_part, domain = email.split("@", 1)
    if len(local_part) <= 2:
        masked_local = local_part[:1] + "*"
    else:
        masked_local = local_part[:2] + "***" + local_part[-1:]
    return f"{masked_local}@{domain}"


def _build_session_cookie_value(user: object) -> str:
    return f"{user['username']}|{str(user['password_hash'])[:24]}"


def _get_authenticated_user(request: Request, settings: AppSettings) -> object | None:
    cookie_value = request.cookies.get(settings.session_cookie_name)
    if not cookie_value or "|" not in cookie_value:
        return None
    username, fingerprint = cookie_value.split("|", 1)
    if not username:
        return None
    user = get_user(settings.db_path, username)
    if user is None:
        return None
    if not str(user["password_hash"]).startswith(fingerprint):
        return None
    return user


def _serialize_latest_analysis_row(latest_analysis_row: object) -> dict[str, object]:
    return {
        "symbol": latest_analysis_row["symbol"],
        "provider": latest_analysis_row["analysis_provider"],
        "model_name": latest_analysis_row["model_name"],
        "status": latest_analysis_row["status"],
        "error_message": latest_analysis_row["error_message"],
        "created_at": latest_analysis_row["created_at"],
        "analysis": json.loads(latest_analysis_row["analysis_json"]),
    }


def _send_trade_analysis_email_if_configured(db_path: str, owner_username: str, symbol: str) -> str | None:
    latest_analysis_row = get_latest_trade_analysis(db_path, owner_username, symbol)
    if latest_analysis_row is None:
        return None

    latest_analysis = _serialize_latest_analysis_row(latest_analysis_row)
    email_settings = get_email_settings(db_path, owner_username)
    result = send_message(
        email_settings,
        subject=f"[交易复盘] {symbol} {latest_analysis['analysis']['judgment']}",
        body=build_trade_analysis_email_body(latest_analysis),
    )
    if result.success:
        return "复盘结果邮件已发送"
    return f"复盘结果邮件发送失败：{result.error}"


def _require_login(request: Request, settings: AppSettings) -> object | RedirectResponse:
    current_user = _get_authenticated_user(request, settings)
    if current_user is not None:
        return current_user
    return RedirectResponse(url="/login", status_code=303)


def _redirect_with_message(path: str, message: str) -> RedirectResponse:
    return RedirectResponse(url=f"{path}?message={quote(message)}", status_code=303)


def _login_subject(request: Request, username: str = "") -> str:
    forwarded_for = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    client_host = forwarded_for or (request.client.host if request.client else "unknown")
    user_agent = (request.headers.get("user-agent") or "unknown")[:120]
    return f"{username.lower()}|{client_host}|{user_agent}"


async def _load_dashboard_snapshots(app: FastAPI, owner_username: str) -> list[object]:
    settings: AppSettings = app.state.settings
    monitor: StockMonitor = app.state.monitor
    snapshots = get_snapshots(settings.db_path, owner_username)
    missing_symbols = [item["symbol"] for item in snapshots if item["latest_price"] is None]
    if not missing_symbols:
        return snapshots

    for symbol in missing_symbols:
        try:
            await asyncio.to_thread(monitor.refresh_symbol_snapshot, owner_username, symbol)
        except Exception:
            continue
    return get_snapshots(settings.db_path, owner_username)


def run() -> None:
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
