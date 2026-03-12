from datetime import date, datetime
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.ai_stock_monitoring.app import create_app
from src.ai_stock_monitoring.config import AppSettings
from src.ai_stock_monitoring.database import initialize_database
from src.ai_stock_monitoring.market_hours import TradeCalendar, get_market_status
from src.ai_stock_monitoring.monitor import (
    StockMonitor,
    calculate_simple_moving_average,
    has_weekly_crossed,
    parse_stock_symbols,
    validate_stock_symbol,
)
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
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                dashboard_response = client.get("/dashboard")
                self.assertEqual(dashboard_response.status_code, 200)
                self.assertIn("监控状态", dashboard_response.text)

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
                    send_response = client.post(
                        "/trades/email-analysis",
                        data={"symbol": "600519"},
                        follow_redirects=False,
                    )
                    self.assertEqual(send_response.status_code, 303)
                    mocked_send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
