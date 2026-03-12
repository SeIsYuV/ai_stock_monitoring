from datetime import date, datetime
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.ai_stock_monitoring.app import create_app
from src.ai_stock_monitoring.config import AppSettings
from src.ai_stock_monitoring.database import get_login_unlock_code, get_user, initialize_database
from src.ai_stock_monitoring.market_hours import TradeCalendar, get_market_status
from src.ai_stock_monitoring.monitor import (
    StockMonitor,
    calculate_simple_moving_average,
    has_weekly_crossed,
    parse_stock_symbols,
    validate_stock_symbol,
)
from src.ai_stock_monitoring.quant import build_quant_signal
from src.ai_stock_monitoring.security import verify_password
from src.ai_stock_monitoring.trade_advisor import build_position_summary


class MonitorTests(unittest.TestCase):
    def test_validate_stock_symbol(self) -> None:
        self.assertTrue(validate_stock_symbol("600519"))
        self.assertFalse(validate_stock_symbol("A00519"))
        self.assertFalse(validate_stock_symbol("12345"))

    def test_parse_stock_symbols_supports_batch_input(self) -> None:
        valid_symbols, invalid_symbols = parse_stock_symbols("600519\n000001,abc 300750")
        self.assertEqual(valid_symbols, ["600519", "000001", "300750"])
        self.assertEqual(invalid_symbols, ["abc"])

    def test_moving_average_and_cross(self) -> None:
        self.assertEqual(calculate_simple_moving_average([1, 2, 3, 4], 2), 3.5)
        self.assertTrue(has_weekly_crossed(11.0, 10.0, 9.5, 10.0))

    def test_market_status_uses_trade_calendar(self) -> None:
        status = get_market_status(
            refresh_interval_seconds=30,
            trade_calendar=TradeCalendar([date(2026, 3, 12)]),
            now=datetime.fromisoformat("2026-03-12T10:00:00+08:00"),
        )
        self.assertTrue(status.is_market_open)
        self.assertEqual(status.label, "监控中（A股 9:30-15:00）")

    def test_monitor_summary(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock")
            initialize_database(settings)
            summary = StockMonitor(settings).run()
            self.assertIn("Monitoring 3 stock(s)", summary)

    def test_dashboard_route_renders(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                login_response = client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                self.assertEqual(login_response.status_code, 303)
                dashboard_response = client.get("/dashboard")
                self.assertEqual(dashboard_response.status_code, 200)
                self.assertIn("监控状态", dashboard_response.text)
                self.assertIn("当前账号：admin", dashboard_response.text)
                self.assertIn("dashboard-current-time", dashboard_response.text)
                self.assertIn("dashboard-refresh-countdown", dashboard_response.text)

    def test_login_page_hides_default_credentials(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                response = client.get("/login")
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("admin123", response.text)
                self.assertNotIn("默认账号", response.text)
                self.assertIn("发送解封验证码", response.text)

    def test_build_position_summary(self) -> None:
        summary = build_position_summary(
            [
                {"side": "buy", "price": 10.0, "quantity": 100},
                {"side": "buy", "price": 12.0, "quantity": 100},
                {"side": "sell", "price": 15.0, "quantity": 50},
            ]
        )
        self.assertEqual(summary["position_quantity"], 150)
        self.assertAlmostEqual(summary["average_cost"], 11.0, places=2)
        self.assertAlmostEqual(summary["realized_pnl"], 200.0, places=2)

    def test_trades_analysis_route_renders(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                create_response = client.post(
                    "/trades",
                    data={
                        "symbol": "600519",
                        "side": "buy",
                        "price": "1200.5",
                        "quantity": "100",
                        "traded_at": "2026-03-12T10:00",
                        "note": "首次建仓",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(create_response.status_code, 303)

                analyze_response = client.post(
                    "/trades/analyze",
                    data={"symbol": "600519"},
                    follow_redirects=False,
                )
                self.assertEqual(analyze_response.status_code, 303)

                trades_page = client.get("/trades?symbol=600519")
                self.assertEqual(trades_page.status_code, 200)
                self.assertIn("最新分析", trades_page.text)
                self.assertIn("首次建仓", trades_page.text)

    def test_trade_export_route_returns_excel(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                client.post(
                    "/trades",
                    data={
                        "symbol": "600519",
                        "side": "buy",
                        "price": "1200.5",
                        "quantity": "100",
                        "traded_at": "2026-03-12T10:00",
                        "note": "首次建仓",
                    },
                    follow_redirects=False,
                )
                export_response = client.get("/trades/export?symbol=600519")
                self.assertEqual(export_response.status_code, 200)
                self.assertEqual(
                    export_response.headers["content-type"],
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    def test_trade_analysis_email_route_sends_mail(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                client.post(
                    "/settings/email",
                    data={
                        "recipient_email": "to@example.com",
                        "smtp_server": "smtp.qq.com",
                        "sender_email": "from@example.com",
                        "sender_password": "secret",
                    },
                    follow_redirects=False,
                )
                client.post(
                    "/trades",
                    data={
                        "symbol": "600519",
                        "side": "buy",
                        "price": "1200.5",
                        "quantity": "100",
                        "traded_at": "2026-03-12T10:00",
                        "note": "首次建仓",
                    },
                    follow_redirects=False,
                )
                client.post(
                    "/trades/analyze",
                    data={"symbol": "600519"},
                    follow_redirects=False,
                )
                with patch("src.ai_stock_monitoring.app.send_message") as mocked_send:
                    mocked_send.return_value.success = True
                    mocked_send.return_value.error = None
                    mocked_send.return_value.status = "success"
                    send_response = client.post(
                        "/trades/email-analysis",
                        data={"symbol": "600519"},
                        follow_redirects=False,
                    )
                    self.assertEqual(send_response.status_code, 303)
                    mocked_send.assert_called_once()

    def test_add_stock_refreshes_snapshot_immediately(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                response = client.post(
                    "/stocks",
                    data={"symbols_text": "600111"},
                    follow_redirects=False,
                )
                self.assertEqual(response.status_code, 303)

                dashboard_response = client.get("/dashboard")
                self.assertEqual(dashboard_response.status_code, 200)
                self.assertIn("600111", dashboard_response.text)
                self.assertIn("示例股票600111", dashboard_response.text)

    def test_account_password_change_persists_after_restart(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock")
            app = create_app(settings)
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                change_response = client.post(
                    "/settings/admin-password",
                    data={
                        "current_password": "admin123",
                        "new_password": "betterpass123",
                        "confirm_password": "betterpass123",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(change_response.status_code, 303)

            restarted_app = create_app(settings)
            with TestClient(restarted_app) as restarted_client:
                success_response = restarted_client.post(
                    "/login",
                    data={"username": "admin", "password": "betterpass123"},
                    follow_redirects=False,
                )
                self.assertEqual(success_response.status_code, 303)
                failed_response = restarted_client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                self.assertEqual(failed_response.status_code, 400)

    def test_login_guard_locks_after_repeated_failures(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                for _ in range(5):
                    response = client.post(
                        "/login",
                        data={"username": "admin", "password": "wrong-password"},
                        follow_redirects=False,
                    )
                    self.assertEqual(response.status_code, 400)

                locked_response = client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                self.assertEqual(locked_response.status_code, 429)
                self.assertIn("临时锁定", locked_response.text)

    def test_password_helper_supports_legacy_hashes(self) -> None:
        self.assertTrue(verify_password("admin123", "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"))

    def test_admin_can_create_user_and_data_is_isolated(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as admin_client:
                admin_client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                create_user_response = admin_client.post(
                    "/settings/users",
                    data={
                        "username": "alice",
                        "password": "alicepass123",
                        "confirm_password": "alicepass123",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(create_user_response.status_code, 303)
                admin_client.post(
                    "/stocks",
                    data={"symbols_text": "600519"},
                    follow_redirects=False,
                )
                admin_dashboard = admin_client.get("/dashboard")
                self.assertIn("600519", admin_dashboard.text)

            with TestClient(app) as alice_client:
                login_response = alice_client.post(
                    "/login",
                    data={"username": "alice", "password": "alicepass123"},
                    follow_redirects=False,
                )
                self.assertEqual(login_response.status_code, 303)
                alice_client.post(
                    "/stocks",
                    data={"symbols_text": "000001"},
                    follow_redirects=False,
                )
                alice_dashboard = alice_client.get("/dashboard")
                self.assertIn("000001", alice_dashboard.text)
                self.assertNotIn("600519</a></td>", alice_dashboard.text)

    def test_quant_settings_and_alerts_work(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                client.post(
                    "/stocks",
                    data={"symbols_text": "600519"},
                    follow_redirects=False,
                )
                settings_response = client.post(
                    "/settings/quant",
                    data={
                        "enabled": "1",
                        "probability_threshold": "50",
                        "selected_models": [
                            "trend_following",
                            "mean_reversion",
                            "dividend_quality",
                            "weekly_resonance",
                        ],
                    },
                    follow_redirects=False,
                )
                self.assertEqual(settings_response.status_code, 303)
                app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-12T10:00:00+08:00"))
                dashboard = client.get("/dashboard")
                self.assertIn("量化盈利概率", dashboard.text)
                history = client.get("/history")
                self.assertIn("量化盈利概率", history.text)

    def test_quant_signal_returns_probability(self) -> None:
        settings = AppSettings(provider_name="mock")
        monitor = StockMonitor(settings)
        bars_daily = monitor.provider.get_daily_bars("600519")
        bars_weekly = monitor.provider.get_weekly_bars("600519")
        signal = build_quant_signal(
            latest_price=monitor.provider.get_quote("600519").latest_price,
            ma_250=calculate_simple_moving_average([item.close_price for item in bars_daily], 250),
            boll_mid=calculate_simple_moving_average([item.close_price for item in bars_daily], 20),
            ma_30w=calculate_simple_moving_average([item.close_price for item in bars_weekly], 30),
            ma_60w=calculate_simple_moving_average([item.close_price for item in bars_weekly], 60),
            dividend_yield=monitor.provider.get_trailing_dividend_yield("600519", monitor.provider.get_quote("600519").latest_price),
            daily_bars=bars_daily,
            weekly_bars=bars_weekly,
        )
        self.assertGreaterEqual(signal.probability, 1)
        self.assertLessEqual(signal.probability, 99)
        self.assertIn("模型", signal.summary)

    def test_created_user_password_is_stored(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                client.post(
                    "/settings/users",
                    data={
                        "username": "bob",
                        "password": "bob-pass-123",
                        "confirm_password": "bob-pass-123",
                    },
                    follow_redirects=False,
                )
            user = get_user(database_file.name, "bob")
            self.assertIsNotNone(user)
            assert user is not None
            self.assertTrue(verify_password("bob-pass-123", user["password_hash"]))

    def test_email_unlock_flow_clears_login_guard(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                client.post(
                    "/settings/email",
                    data={
                        "recipient_email": "to@example.com",
                        "smtp_server": "smtp.qq.com",
                        "sender_email": "from@example.com",
                        "sender_password": "secret",
                    },
                    follow_redirects=False,
                )
                client.get("/logout", follow_redirects=False)

                for _ in range(5):
                    response = client.post(
                        "/login",
                        data={"username": "admin", "password": "wrong-password"},
                        follow_redirects=False,
                    )
                    self.assertEqual(response.status_code, 400)

                locked_response = client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                self.assertEqual(locked_response.status_code, 429)

                with patch("src.ai_stock_monitoring.app.send_message") as mocked_send:
                    mocked_send.return_value.success = True
                    mocked_send.return_value.status = "发送成功"
                    mocked_send.return_value.error = None
                    unlock_request = client.post(
                        "/login/unlock/request",
                        data={"username": "admin"},
                        follow_redirects=False,
                    )
                    self.assertEqual(unlock_request.status_code, 200)
                    self.assertIn("验证码已发送", unlock_request.text)

                unlock_row = get_login_unlock_code(database_file.name, "admin")
                self.assertIsNotNone(unlock_row)
                assert unlock_row is not None
                confirm_response = client.post(
                    "/login/unlock/confirm",
                    data={"username": "admin", "verification_code": unlock_row["verification_code"]},
                    follow_redirects=False,
                )
                self.assertEqual(confirm_response.status_code, 200)
                self.assertIn("账号锁定已解除", confirm_response.text)

                login_response = client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                self.assertEqual(login_response.status_code, 303)



if __name__ == "__main__":
    unittest.main()
