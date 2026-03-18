from datetime import date, datetime, timedelta
import json
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.ai_stock_monitoring.app import _build_model_review_payload, _format_snapshot_timestamp, create_app
from src.ai_stock_monitoring.config import AppSettings
from src.ai_stock_monitoring.database import add_trade_record, get_alert_history, get_login_unlock_code, get_snapshot, get_user, initialize_database, list_model_paper_trades, list_recent_login_events, list_trade_records, upsert_snapshot
from src.ai_stock_monitoring.mailer import build_alert_email_body, build_alert_email_html_body, build_portfolio_review_email_body
from src.ai_stock_monitoring.market_hours import TradeCalendar, get_market_status
from src.ai_stock_monitoring.monitor import (
    SnapshotComputation,
    StockMonitor,
    build_quant_sell_signal,
    calculate_bollinger_lower_band,
    calculate_bollinger_upper_band,
    calculate_simple_moving_average,
    get_weekly_cross_direction,
    has_effective_breakout_above_ma60w,
    has_weekly_crossed,
    parse_stock_symbols,
    validate_stock_symbol,
)
from src.ai_stock_monitoring.quant import build_quant_signal
from src.ai_stock_monitoring.providers.base import MarketDataProvider, PriceBar, Quote
from src.ai_stock_monitoring.security import verify_password
from src.ai_stock_monitoring.trade_advisor import build_market_action_summary, build_portfolio_profile, build_position_summary, build_recommended_price_plan, build_stock_comprehensive_advice


class RefreshAwareProvider(MarketDataProvider):
    provider_name = "refresh-aware"

    def __init__(self) -> None:
        self.version = 0
        self.invalidated_symbols: list[str] = []

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            name=f"测试股票{symbol}",
            latest_price=10.0 + self.version,
            updated_at=datetime.fromisoformat("2026-03-18T10:00:00+00:00"),
        )

    def get_daily_bars(self, symbol: str, limit: int = 320) -> list[PriceBar]:
        base = 10.0 + self.version
        start = date(2025, 1, 1)
        return [
            PriceBar(
                traded_on=start + timedelta(days=index),
                open_price=base - 0.2,
                high_price=base + 0.4,
                low_price=base - 0.4,
                close_price=base + (index % 6) * 0.03,
                volume=1_000_000 + index * 1000,
            )
            for index in range(limit)
        ]

    def get_weekly_bars(self, symbol: str, limit: int = 80) -> list[PriceBar]:
        base = 10.0 + self.version
        start = date(2024, 1, 5)
        return [
            PriceBar(
                traded_on=start + timedelta(days=index * 7),
                open_price=base - 0.3,
                high_price=base + 0.5,
                low_price=base - 0.5,
                close_price=base + (index % 5) * 0.08,
                volume=5_000_000 + index * 5000,
            )
            for index in range(limit)
        ]

    def get_trailing_dividend_yield(self, symbol: str, latest_price: float) -> float:
        return 4.2

    def get_trade_dates(self) -> list[date]:
        return [date(2026, 3, 18)]

    def invalidate_symbol_cache(self, symbol: str) -> None:
        self.invalidated_symbols.append(symbol)
        self.version += 1


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
        self.assertEqual(get_weekly_cross_direction(9.5, 10.0, 10.5, 10.0), "bullish")
        self.assertEqual(get_weekly_cross_direction(10.5, 10.0, 9.5, 10.0), "bearish")

    def test_effective_breakout_above_ma60w_requires_close_and_low_confirmation(self) -> None:
        weekly_bars = [
            PriceBar(date(2026, 3, 6), 99.0, 101.0, 98.0, 100.0),
            PriceBar(date(2026, 3, 13), 100.5, 104.0, 101.5, 103.0),
        ]
        self.assertTrue(has_effective_breakout_above_ma60w(weekly_bars, current_ma_60w=101.0, prev_ma_60w=100.5))
        weak_breakout = [
            weekly_bars[0],
            PriceBar(date(2026, 3, 13), 100.5, 104.0, 99.8, 101.2),
        ]
        self.assertFalse(has_effective_breakout_above_ma60w(weak_breakout, current_ma_60w=101.0, prev_ma_60w=100.5))

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

    def test_refresh_symbol_snapshot_force_refresh_invalidates_provider_cache(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock")
            initialize_database(settings)
            monitor = StockMonitor(settings)
            monitor.provider = RefreshAwareProvider()
            normal_snapshot = monitor.refresh_symbol_snapshot("admin", "600519")
            forced_snapshot = monitor.refresh_symbol_snapshot("admin", "600519", force_refresh=True)
            self.assertEqual(normal_snapshot.latest_price, 10.0)
            self.assertEqual(forced_snapshot.latest_price, 11.0)
            self.assertEqual(monitor.provider.invalidated_symbols, ["600519"])

    def test_model_paper_trades_open_and_close_from_snapshot_scores(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock")
            initialize_database(settings)
            monitor = StockMonitor(settings)
            open_snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=100.0,
                ma_250=95.0,
                ma_30w=96.0,
                ma_60w=92.0,
                prev_ma_30w=95.0,
                prev_ma_60w=91.0,
                boll_mid=98.0,
                boll_lower=94.0,
                boll_upper=108.0,
                dividend_yield=3.8,
                quant_probability=78.0,
                quant_model_breakdown=json.dumps(
                    [
                        {"key": "msci_momentum", "label": "MSCI动量", "score": 80},
                        {"key": "quality_stability", "label": "质量稳定", "score": 76},
                        {"key": "trend_following", "label": "趋势跟随", "score": 74},
                    ],
                    ensure_ascii=False,
                ),
                trigger_state="量化盈利概率",
                trigger_detail="test",
                triggered_labels=("量化盈利概率",),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-18T10:00:00+00:00"),
            )
            monitor._sync_model_paper_trades("admin", open_snapshot)
            opened = list_model_paper_trades(database_file.name, "admin")
            self.assertTrue(opened)
            self.assertTrue(any(row["status"] == "open" for row in opened))

            close_snapshot = SnapshotComputation(
                **{
                    **open_snapshot.__dict__,
                    "latest_price": 96.0,
                    "quant_model_breakdown": json.dumps(
                        [
                            {"key": "msci_momentum", "label": "MSCI动量", "score": 42},
                            {"key": "quality_stability", "label": "质量稳定", "score": 44},
                            {"key": "trend_following", "label": "趋势跟随", "score": 40},
                        ],
                        ensure_ascii=False,
                    ),
                    "updated_at": datetime.fromisoformat("2026-03-25T10:00:00+00:00"),
                }
            )
            monitor._sync_model_paper_trades("admin", close_snapshot)
            closed = list_model_paper_trades(database_file.name, "admin")
            self.assertTrue(any(row["status"] == "closed" for row in closed))

    def test_model_paper_trades_support_take_profit_stop_loss_and_timeout_exits(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock")
            initialize_database(settings)
            monitor = StockMonitor(settings)

            def build_snapshot(symbol: str, latest_price: float, score: float, updated_at: str) -> SnapshotComputation:
                return SnapshotComputation(
                    symbol=symbol,
                    display_name=f"测试{symbol}",
                    latest_price=latest_price,
                    ma_250=95.0,
                    ma_30w=96.0,
                    ma_60w=92.0,
                    prev_ma_30w=95.0,
                    prev_ma_60w=91.0,
                    boll_mid=98.0,
                    boll_lower=94.0,
                    boll_upper=108.0,
                    dividend_yield=3.8,
                    quant_probability=78.0,
                    quant_model_breakdown=json.dumps(
                        [
                            {"key": "trend_following", "label": "趋势跟随", "score": score},
                        ],
                        ensure_ascii=False,
                    ),
                    trigger_state="量化盈利概率",
                    trigger_detail="test",
                    triggered_labels=("量化盈利概率",),
                    weekly_crossed=False,
                    updated_at=datetime.fromisoformat(updated_at),
                )

            monitor._sync_model_paper_trades("admin", build_snapshot("600519", 100.0, 82.0, "2026-03-01T10:00:00+00:00"))
            monitor._sync_model_paper_trades("admin", build_snapshot("600519", 113.5, 88.0, "2026-03-05T10:00:00+00:00"))
            take_profit_rows = list_model_paper_trades(database_file.name, "admin", symbol="600519", status="closed")
            self.assertTrue(any("止盈退出" in str(row["exit_reason"]) for row in take_profit_rows))

            monitor._sync_model_paper_trades("admin", build_snapshot("601318", 100.0, 82.0, "2026-03-01T10:00:00+00:00"))
            monitor._sync_model_paper_trades("admin", build_snapshot("601318", 91.0, 82.0, "2026-03-03T10:00:00+00:00"))
            stop_loss_rows = list_model_paper_trades(database_file.name, "admin", symbol="601318", status="closed")
            self.assertTrue(any("止损退出" in str(row["exit_reason"]) for row in stop_loss_rows))

            monitor._sync_model_paper_trades("admin", build_snapshot("600036", 100.0, 82.0, "2026-03-01T10:00:00+00:00"))
            monitor._sync_model_paper_trades("admin", build_snapshot("600036", 102.0, 84.0, "2026-03-20T10:00:00+00:00"))
            timeout_rows = list_model_paper_trades(database_file.name, "admin", symbol="600036", status="closed")
            self.assertTrue(any("超期退出" in str(row["exit_reason"]) for row in timeout_rows))

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
                self.assertIn("账户仓位总览", dashboard_response.text)
                self.assertIn("DCF内在价值/偏差", dashboard_response.text)
                self.assertIn("综合建议", dashboard_response.text)
                self.assertTrue("买入｜" in dashboard_response.text or "卖出｜" in dashboard_response.text)
                self.assertIn("买点", dashboard_response.text)
                self.assertIn("卖点", dashboard_response.text)
                self.assertIn("大盘", dashboard_response.text)
                self.assertIn("量能比", dashboard_response.text)
                self.assertIn("dashboard-current-time", dashboard_response.text)
                self.assertIn("dashboard-refresh-countdown", dashboard_response.text)

    def test_model_review_route_renders(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                login_response = client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                self.assertEqual(login_response.status_code, 303)
                response = client.get("/model-review")
                self.assertEqual(response.status_code, 200)
                self.assertIn("模型复盘", response.text)
                self.assertIn("月度净值与回撤", response.text)
                self.assertIn("离场规则统计", response.text)
                self.assertIn("当前月份暂无纸面交易记录", response.text)

    def test_model_review_payload_includes_curve_and_exit_stats(self) -> None:
        payload = _build_model_review_payload(
            [
                {
                    "model_key": "professional",
                    "model_label": "专业组",
                    "model_scope": "group",
                    "symbol": "600519",
                    "display_name": "贵州茅台",
                    "status": "closed",
                    "entry_price": 100.0,
                    "latest_price": 112.0,
                    "exit_price": 112.0,
                    "entry_date": "2026-03-02",
                    "latest_date": "2026-03-08",
                    "exit_date": "2026-03-08",
                    "holding_days": 6,
                    "max_return_pct": 13.0,
                    "max_drawdown_pct": 4.2,
                    "realized_return_pct": 12.0,
                    "unrealized_return_pct": 0.0,
                    "entry_reason": "test",
                    "exit_reason": "止盈退出：当前收益 12.00% ，达到止盈阈值 12.00% 。",
                },
                {
                    "model_key": "adaptive",
                    "model_label": "自适应组",
                    "model_scope": "group",
                    "symbol": "601318",
                    "display_name": "中国平安",
                    "status": "closed",
                    "entry_price": 100.0,
                    "latest_price": 92.0,
                    "exit_price": 92.0,
                    "entry_date": "2026-03-10",
                    "latest_date": "2026-03-16",
                    "exit_date": "2026-03-16",
                    "holding_days": 6,
                    "max_return_pct": 2.0,
                    "max_drawdown_pct": 8.5,
                    "realized_return_pct": -8.0,
                    "unrealized_return_pct": 0.0,
                    "entry_reason": "test",
                    "exit_reason": "止损退出：当前收益 -8.00% ，触发止损阈值 -8.00% 。",
                },
            ],
            month_start=date(2026, 3, 1),
            month_end=date(2026, 3, 31),
        )
        self.assertTrue(payload["monthly_equity"]["net_value_chart"]["points"])
        self.assertTrue(payload["monthly_equity"]["drawdown_chart"]["points"])
        self.assertEqual(payload["exit_summary_rows"][0]["exit_label"], "止盈退出")
        self.assertTrue(any(item["exit_label"] == "止损退出" for item in payload["exit_summary_rows"]))

    def test_login_page_hides_default_credentials(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                response = client.get("/login")
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("admin123", response.text)
                self.assertNotIn("默认账号", response.text)
                self.assertIn("发送解封验证码", response.text)

    def test_add_trade_record_rounds_price_to_five_decimals(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            initialize_database(AppSettings(db_path=database_file.name, provider_name="mock"))
            add_trade_record(
                database_file.name,
                owner_username="admin",
                symbol="600519",
                side="buy",
                price=12.3456789,
                quantity=100,
                note="test",
            )
            records = list_trade_records(database_file.name, "admin", "600519")
            self.assertEqual(len(records), 1)
            self.assertAlmostEqual(float(records[0]["price"]), 12.34568, places=5)
            self.assertTrue(str(records[0]["traded_at"]))


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


    def test_build_recommended_price_plan_contains_explicit_ranges(self) -> None:
        signal_summary = build_market_action_summary(
            {
                "trigger_state": "250日线、BOLL下轨、量化盈利概率",
                "quant_probability": 91,
                "quant_model_breakdown": '[{"label": "趋势跟随", "score": 93}]',
            }
        )
        plan = build_recommended_price_plan(
            {
                "latest_price": 10.0,
                "ma_250": 9.8,
                "boll_mid": 10.2,
                "boll_lower": 9.5,
                "boll_upper": 11.4,
            },
            signal_summary,
            {"average_cost": 9.2},
        )
        self.assertIn("-", plan["recommended_buy_price_range"])
        self.assertIn("-", plan["recommended_sell_price_range"])
        self.assertGreater(plan["suggested_add_price"], 0)
        self.assertGreater(plan["suggested_reduce_price"], 0)
        self.assertGreater(plan["suggested_stop_loss_price"], 0)
        self.assertTrue(plan["buy_price_plan"])
        self.assertTrue(plan["watch_price_plan"])
        watch_left, watch_right = [float(item.strip()) for item in plan["watch_price_range"].split("-")]
        self.assertLess(watch_right - watch_left, 0.2)


    def test_build_portfolio_profile_summarizes_holdings(self) -> None:
        profile = build_portfolio_profile(
            [
                {"symbol": "600519", "side": "buy", "price": 1000.0, "quantity": 100},
                {"symbol": "600519", "side": "buy", "price": 1100.0, "quantity": 100},
                {"symbol": "000001", "side": "buy", "price": 10.0, "quantity": 1000},
                {"symbol": "000001", "side": "sell", "price": 12.0, "quantity": 400},
            ],
            [
                {
                    "symbol": "600519",
                    "display_name": "贵州茅台",
                    "latest_price": 1180.0,
                    "ma_250": 1120.0,
                    "boll_mid": 1160.0,
                    "boll_lower": 1090.0,
                    "boll_upper": 1250.0,
                    "dividend_yield": 3.2,
                    "quant_probability": 82.0,
                    "quant_model_breakdown": '[{"label": "趋势跟随", "score": 88}]',
                    "trigger_state": "250日线、量化盈利概率",
                },
                {
                    "symbol": "000001",
                    "display_name": "平安银行",
                    "latest_price": 11.2,
                    "ma_250": 10.8,
                    "boll_mid": 11.0,
                    "boll_lower": 10.5,
                    "boll_upper": 11.8,
                    "dividend_yield": 5.1,
                    "quant_probability": 68.0,
                    "quant_model_breakdown": '[{"label": "股息质量", "score": 79}]',
                    "trigger_state": "股息率",
                },
            ],
        )
        self.assertTrue(profile["has_positions"])
        self.assertGreater(profile["holding_ratio"], 0)
        self.assertIn("%", profile["recommended_holding_ratio"])
        self.assertEqual(len(profile["active_positions"]), 2)
        self.assertGreater(profile["active_positions"][0]["suggested_add_price"], 0)
        self.assertGreater(profile["active_positions"][0]["suggested_reduce_price"], 0)
        self.assertGreater(profile["active_positions"][0]["suggested_stop_loss_price"], 0)
        self.assertIn("buy_recommendation_level", profile["active_positions"][0])
        self.assertIn("sell_recommendation_level", profile["active_positions"][0])
        self.assertIn("recommended_buy_price_range", profile["active_positions"][0])
        self.assertIn("recommended_sell_price_range", profile["active_positions"][0])
        self.assertTrue(profile["overall_adjustment_suggestions"])
        self.assertIn(profile["risk_level"], {"低", "中", "高"})
        self.assertTrue(profile["comprehensive_advice"])
        self.assertTrue(profile["active_positions"][0]["comprehensive_advice"])

    def test_build_stock_comprehensive_advice_includes_prices_and_dcf_reason(self) -> None:
        advice = build_stock_comprehensive_advice(
            {
                "symbol": "600000",
                "latest_price": 10.5,
                "ma_250": 10.1,
                "boll_mid": 10.3,
                "boll_lower": 9.9,
                "boll_upper": 11.1,
                "quant_probability": 76.0,
                "quant_model_breakdown": json.dumps(
                    [
                        {"key": "trend_following", "label": "趋势跟随", "score": 88},
                        {"key": "weekly_resonance", "label": "周线共振", "score": 83},
                        {"key": "support_strength", "label": "支撑强度", "score": 79},
                        {"key": "risk_reward", "label": "盈亏比", "score": 75},
                        {"key": "dcf_proxy", "label": "DCF估值", "score": 35, "reason": "缺少稳定现金流代理数据，DCF 估值置信度较低"},
                    ],
                    ensure_ascii=False,
                ),
                "trigger_state": "250日线、量化盈利概率",
            }
        )
        self.assertIn("买点：", advice["comprehensive_advice"])
        self.assertIn("卖点：", advice["comprehensive_advice"])
        self.assertIn("买入等级", advice["comprehensive_advice"])
        self.assertIn("卖出等级", advice["comprehensive_advice"])
        self.assertIn("四模型综合", advice["comprehensive_advice"])
        self.assertGreaterEqual(advice["buy_recommendation_level"], 1)
        self.assertLessEqual(advice["buy_recommendation_level"], 10)
        self.assertGreaterEqual(advice["sell_recommendation_level"], 1)
        self.assertLessEqual(advice["sell_recommendation_level"], 10)
        self.assertIsNone(advice["dcf_intrinsic_value"])
        self.assertIn("DCF", advice["dcf_reason"])

    def test_enhanced_quant_models_are_in_snapshot_breakdown(self) -> None:
        monitor = StockMonitor(AppSettings(provider_name="mock"))
        snapshot = monitor.build_snapshot("600519")
        labels = {item["label"] for item in json.loads(snapshot.quant_model_breakdown)}
        self.assertIn("支撑强度", labels)
        self.assertIn("盈亏比", labels)
        self.assertIn("DCF估值", labels)

    def test_idle_session_redirects_to_login(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock")
            app = create_app(settings)
            with TestClient(app) as client:
                user = get_user(database_file.name, "admin")
                self.assertIsNotNone(user)
                stale_timestamp = 1
                client.cookies.set(
                    settings.session_cookie_name,
                    f"admin|{str(user['password_hash'])[:24]}|{stale_timestamp}",
                )
                response = client.get("/dashboard", follow_redirects=False)
                self.assertEqual(response.status_code, 303)
                self.assertIn("/login?message=", response.headers["location"])

    def test_stock_detail_page_renders_dcf_metrics(self) -> None:
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
                app.state.monitor.refresh_symbol_snapshot("admin", "600519")
                response = client.get("/stocks/600519")
                self.assertEqual(response.status_code, 200)
                self.assertIn("DCF 代理内在价值", response.text)
                self.assertIn("大盘环境", response.text)
                self.assertIn("财报节奏", response.text)

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
                self.assertIn("当前账户持仓画像", trades_page.text)
                self.assertIn("模型建议仓位", trades_page.text)
                self.assertIn("建议加仓价", trades_page.text)
                self.assertIn("建议止损价", trades_page.text)
                self.assertIn("最新分析", trades_page.text)
                self.assertIn("首次建仓", trades_page.text)
                self.assertIn("综合建议", trades_page.text)
                self.assertIn("DCF代理内在价值", trades_page.text)
                self.assertIn("推荐买入价", trades_page.text)
                self.assertIn("观望关注价", trades_page.text)
                self.assertIn("买入等级", trades_page.text)
                self.assertIn("卖出等级", trades_page.text)

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

    def test_email_settings_test_route_sends_mail(self) -> None:
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
                with patch("src.ai_stock_monitoring.app.send_message") as mocked_send:
                    mocked_send.return_value.success = True
                    mocked_send.return_value.error = None
                    mocked_send.return_value.status = "发送成功"
                    send_response = client.post(
                        "/settings/email/test",
                        follow_redirects=False,
                    )
                    self.assertEqual(send_response.status_code, 303)
                    mocked_send.assert_called_once()
                    self.assertIn("[邮箱测试]", mocked_send.call_args.kwargs["subject"])

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

    def test_boll_upper_band_is_available_in_snapshot_and_dashboard(self) -> None:
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
                self.assertIn("周BOLL上/中/下轨", dashboard_response.text)

                chart_payload = app.state.monitor.build_chart_payload("600519")
                self.assertIn("bollUpper", chart_payload)
                self.assertEqual(len(chart_payload["bollUpper"]), len(chart_payload["labels"]))

                bars_daily = app.state.monitor.provider.get_daily_bars("600519")
                upper_band = calculate_bollinger_upper_band([item.close_price for item in bars_daily], 20)
                self.assertGreater(upper_band, 0)

    def test_boll_lower_band_is_available_in_snapshot_and_dashboard(self) -> None:
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
                self.assertIn("周BOLL上/中/下轨", dashboard_response.text)

                chart_payload = app.state.monitor.build_chart_payload("600519")
                self.assertIn("bollLower", chart_payload)
                self.assertEqual(len(chart_payload["bollLower"]), len(chart_payload["labels"]))

                bars_daily = app.state.monitor.provider.get_daily_bars("600519")
                lower_band = calculate_bollinger_lower_band([item.close_price for item in bars_daily], 20)
                self.assertGreater(lower_band, 0)

    def test_dashboard_can_manually_refresh_snapshots(self) -> None:
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
                self.assertIn("手动刷新监控列表", dashboard_response.text)

                refresh_response = client.post("/dashboard/refresh", follow_redirects=False)
                self.assertEqual(refresh_response.status_code, 303)

    def test_dashboard_manual_refresh_keeps_last_snapshot_when_provider_fails(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                original_snapshot = app.state.monitor.refresh_symbol_snapshot("admin", "600519")
                with patch.object(app.state.monitor, "build_snapshot", side_effect=RuntimeError("Connection aborted")):
                    refresh_response = client.post("/dashboard/refresh", follow_redirects=False)
                self.assertEqual(refresh_response.status_code, 303)
                retained_snapshot = get_snapshot(database_file.name, "admin", "600519")
                self.assertIsNotNone(retained_snapshot)
                assert retained_snapshot is not None
                self.assertEqual(float(retained_snapshot["latest_price"]), original_snapshot.latest_price)

    def test_dashboard_backfills_zero_boll_upper_snapshot(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                snapshot = app.state.monitor.refresh_symbol_snapshot("admin", "600519")
                upsert_snapshot(
                    db_path=database_file.name,
                    owner_username="admin",
                    symbol=snapshot.symbol,
                    display_name=snapshot.display_name,
                    latest_price=snapshot.latest_price,
                    ma_250=snapshot.ma_250,
                    ma_30w=snapshot.ma_30w,
                    ma_60w=snapshot.ma_60w,
                    boll_mid=snapshot.boll_mid,
                    boll_lower=snapshot.boll_lower,
                    boll_upper=0.0,
                    dividend_yield=snapshot.dividend_yield,
                    quant_probability=snapshot.quant_probability,
                    quant_model_breakdown=snapshot.quant_model_breakdown,
                    trigger_state=snapshot.trigger_state,
                    trigger_detail=snapshot.trigger_detail,
                    updated_at=snapshot.updated_at.isoformat(),
                )
                zero_snapshot = get_snapshot(database_file.name, "admin", "600519")
                self.assertIsNotNone(zero_snapshot)
                assert zero_snapshot is not None
                self.assertEqual(float(zero_snapshot["boll_upper"]), 0.0)

                dashboard_response = client.get("/dashboard")
                self.assertEqual(dashboard_response.status_code, 200)
                refreshed_snapshot = get_snapshot(database_file.name, "admin", "600519")
                self.assertIsNotNone(refreshed_snapshot)
                assert refreshed_snapshot is not None
                self.assertGreater(float(refreshed_snapshot["boll_upper"]), 0.0)

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
            quant_snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=100.0,
                boll_lower=96.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=92.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 92}, {"label": "周线共振", "score": 88}], ensure_ascii=False),
                trigger_state="250日线、股息率",
                trigger_detail="测试量化提醒-强确认",
                triggered_labels=("250日线", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-12T10:00:00+08:00"),
            )
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
                with patch.object(app.state.monitor, "build_snapshot", return_value=quant_snapshot):
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
            boll_lower=calculate_bollinger_lower_band([item.close_price for item in bars_daily]),
            boll_upper=calculate_bollinger_upper_band([item.close_price for item in bars_daily]),
            ma_30w=calculate_simple_moving_average([item.close_price for item in bars_weekly], 30),
            ma_60w=calculate_simple_moving_average([item.close_price for item in bars_weekly], 60),
            dividend_yield=monitor.provider.get_trailing_dividend_yield("600519", monitor.provider.get_quote("600519").latest_price),
            daily_bars=bars_daily,
            weekly_bars=bars_weekly,
        )
        self.assertGreaterEqual(signal.probability, 1)
        self.assertLessEqual(signal.probability, 99)
        self.assertIn("模型", signal.summary)

    def test_quant_signal_adaptive_learning_adds_backtest_metadata(self) -> None:
        settings = AppSettings(provider_name="mock")
        monitor = StockMonitor(settings)
        bars_daily = monitor.provider.get_daily_bars("600519")
        bars_weekly = monitor.provider.get_weekly_bars("600519")
        quote = monitor.provider.get_quote("600519")
        signal = build_quant_signal(
            latest_price=quote.latest_price,
            ma_250=calculate_simple_moving_average([item.close_price for item in bars_daily], 250),
            boll_mid=calculate_simple_moving_average([item.close_price for item in bars_daily], 20),
            boll_lower=calculate_bollinger_lower_band([item.close_price for item in bars_daily]),
            boll_upper=calculate_bollinger_upper_band([item.close_price for item in bars_daily]),
            ma_30w=calculate_simple_moving_average([item.close_price for item in bars_weekly], 30),
            ma_60w=calculate_simple_moving_average([item.close_price for item in bars_weekly], 60),
            dividend_yield=monitor.provider.get_trailing_dividend_yield("600519", quote.latest_price),
            daily_bars=bars_daily,
            weekly_bars=bars_weekly,
            strategy_params={
                "adaptive_learning_enabled": True,
                "adaptive_lookback_days": 120,
                "adaptive_holding_days": 5,
                "adaptive_min_samples": 4,
                "adaptive_target_return_pct": 0.0,
            },
        )
        self.assertIn("自适应层", signal.summary)
        self.assertTrue(any(item.adaptive_sample_size > 0 for item in signal.models))
        self.assertTrue(any(item.base_score is not None for item in signal.models))
        self.assertIn("MSCI动量", {item.label for item in signal.models})
        self.assertIn("质量稳定", {item.label for item in signal.models})

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

    def test_recent_login_history_is_recorded_and_rendered(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            with TestClient(app) as client:
                login_response = client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    headers={"user-agent": "pytest-agent/1.0", "x-forwarded-for": "1.2.3.4"},
                    follow_redirects=False,
                )
                self.assertEqual(login_response.status_code, 303)
                events = list_recent_login_events(database_file.name, "admin")
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["client_host"], "1.2.3.4")
                self.assertEqual(events[0]["user_agent"], "pytest-agent/1.0")

                settings_response = client.get("/settings")
                self.assertEqual(settings_response.status_code, 200)
                self.assertIn("最近登录记录", settings_response.text)
                self.assertIn("最近一次登录", settings_response.text)
                self.assertIn("账号列表", settings_response.text)
                self.assertIn("最近登录", settings_response.text)
                self.assertIn("1.2.3.4", settings_response.text)
                self.assertIn("20 日最大波动率阈值", settings_response.text)
                self.assertIn("BOLL 中轨最大偏离", settings_response.text)
                self.assertIn("启用自适应学习校准", settings_response.text)
                self.assertIn("专业基准", settings_response.text)
                self.assertIn("table-scroll", settings_response.text)

    def test_market_action_summary_prefers_sell_when_sell_signals_dominate(self) -> None:
        summary = build_market_action_summary(
            {
                "trigger_state": "BOLL上轨卖出、低股息率卖出、量化走弱卖出、250日线",
                "quant_probability": 28,
                "quant_model_breakdown": '[{"label": "趋势跟随", "score": 21}]',
            }
        )
        self.assertEqual(summary["action"], "偏卖出")
        self.assertEqual(summary["action_color"], "sell")
        self.assertIn("卖出信号更强", summary["action_reason"])

    def test_snapshot_timestamp_is_converted_to_shanghai_time(self) -> None:
        today_utc = datetime.now().astimezone().replace(hour=10, minute=42, second=0, microsecond=0)
        expected = today_utc.astimezone().astimezone().strftime("%H:%M")
        sample = today_utc.astimezone().isoformat()
        self.assertEqual(_format_snapshot_timestamp(sample), expected)

    def test_quant_sell_signal_requires_probability_and_multiple_confirmations(self) -> None:
        monitor = StockMonitor(AppSettings(provider_name="mock"))
        snapshot = monitor.build_snapshot("600519")
        weakened_snapshot = snapshot.__class__(
            symbol=snapshot.symbol,
            display_name=snapshot.display_name,
            latest_price=min(snapshot.boll_mid, snapshot.ma_250) - 1 if snapshot.boll_mid and snapshot.ma_250 else snapshot.latest_price,
            ma_250=snapshot.ma_250,
            ma_30w=min(snapshot.ma_30w, snapshot.ma_60w - 0.5) if snapshot.ma_60w else snapshot.ma_30w,
            ma_60w=snapshot.ma_60w,
            prev_ma_30w=snapshot.prev_ma_30w,
            prev_ma_60w=snapshot.prev_ma_60w,
            boll_mid=snapshot.boll_mid,
            boll_lower=snapshot.boll_lower,
            boll_upper=snapshot.boll_upper,
            dividend_yield=snapshot.dividend_yield,
            quant_probability=20.0,
            quant_model_breakdown=snapshot.quant_model_breakdown,
            trigger_state=snapshot.trigger_state,
            trigger_detail=snapshot.trigger_detail,
            triggered_labels=snapshot.triggered_labels,
            weekly_crossed=snapshot.weekly_crossed,
            updated_at=snapshot.updated_at,
        )
        sell_signal = build_quant_sell_signal(weakened_snapshot)
        self.assertTrue(sell_signal.should_alert)
        self.assertGreaterEqual(sell_signal.confirmation_count, 2)
        self.assertIn("量化综合盈利概率", "；".join(sell_signal.reasons))

    def test_quant_settings_redirect_contains_success_message_type(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock"))
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                response = client.post(
                    "/settings/quant",
                    data={
                        "enabled": "1",
                        "probability_threshold": "90",
                        "selected_models": ["trend_following", "mean_reversion"],
                        "min_dividend_yield": "3",
                        "max_20d_volatility_pct": "4",
                        "min_20d_momentum_pct": "1",
                        "max_boll_deviation_pct": "4",
                        "support_zone_tolerance_pct": "3",
                        "min_reward_risk_ratio": "1.8",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(response.status_code, 303)
                self.assertIn("message_type=success", response.headers["location"])

                dashboard = client.get(response.headers["location"])
                self.assertEqual(dashboard.status_code, 200)
                self.assertIn("message-success", dashboard.text)
                self.assertIn("量化提醒设置已保存", dashboard.text)

    def test_quant_alert_is_deduplicated_within_same_day(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            settings = AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=())
            app = create_app(settings)
            quant_snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=100.0,
                boll_lower=96.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=92.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 92}, {"label": "周线共振", "score": 88}], ensure_ascii=False),
                trigger_state="250日线、股息率",
                trigger_detail="测试量化提醒-同日去重",
                triggered_labels=("250日线", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-12T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post(
                    "/login",
                    data={"username": "admin", "password": "admin123"},
                    follow_redirects=False,
                )
                add_stock_response = client.post(
                    "/stocks",
                    data={"symbols_text": "600519"},
                    follow_redirects=False,
                )
                self.assertEqual(add_stock_response.status_code, 303)
                quant_response = client.post(
                    "/settings/quant",
                    data={
                        "enabled": "1",
                        "probability_threshold": "50",
                        "selected_models": [
                            "trend_following",
                            "mean_reversion",
                            "dividend_quality",
                            "weekly_resonance",
                            "volatility_filter",
                        ],
                        "require_price_above_ma250": "1",
                        "require_weekly_bullish": "1",
                        "min_dividend_yield": "3",
                        "max_20d_volatility_pct": "4",
                        "min_20d_momentum_pct": "1",
                        "max_boll_deviation_pct": "4",
                        "support_zone_tolerance_pct": "3",
                        "min_reward_risk_ratio": "1.8",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(quant_response.status_code, 303)

                with patch.object(app.state.monitor, "build_snapshot", return_value=quant_snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-12T10:00:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-12T14:00:00+08:00"))

                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] == "量化盈利概率"
                ]
                self.assertEqual(len(alerts), 1)


    def test_sell_alerts_only_fire_for_held_positions(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            sell_snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=120.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=108.0,
                prev_ma_30w=109.0,
                prev_ma_60w=107.0,
                boll_mid=110.0,
                boll_lower=100.0,
                boll_upper=115.0,
                dividend_yield=2.0,
                quant_probability=65.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 80}], ensure_ascii=False),
                trigger_state="BOLL上轨卖出",
                trigger_detail="测试卖出提醒",
                triggered_labels=("BOLL上轨卖出",),
                weekly_crossed=False,
                weekly_close=120.0,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=sell_snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:30:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] == "BOLL上轨卖出"
                ]
                self.assertEqual(len(alerts), 0)

                add_trade_record(database_file.name, "admin", "600519", "buy", 100.0, 100)
                with patch.object(app.state.monitor, "build_snapshot", return_value=sell_snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-16T10:00:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-16T10:30:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] == "BOLL上轨卖出"
                ]
                self.assertEqual(len(alerts), 1)

    def test_weekly_buy_signals_are_delivered_next_trading_day(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            weekly_snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=97.0,
                ma_250=100.0,
                ma_30w=101.0,
                ma_60w=100.0,
                prev_ma_30w=99.0,
                prev_ma_60w=100.0,
                boll_mid=98.0,
                boll_lower=95.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=93.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 93}, {"label": "周线共振", "score": 89}], ensure_ascii=False),
                trigger_state="250日线、30周线上穿60周线、有效突破60周线",
                trigger_detail="测试周线提醒",
                triggered_labels=("250日线", "30周线上穿60周线", "有效突破60周线"),
                weekly_crossed=True,
                weekly_bullish_crossed=True,
                weekly_breakout_above_ma60w=True,
                weekly_close=103.0,
                prev_weekly_close=99.0,
                updated_at=datetime.fromisoformat("2026-03-13T15:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=weekly_snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T15:10:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-16T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"30周线上穿60周线", "有效突破60周线"}
                ]
                alert_types = [row["trigger_type"] for row in alerts]
                self.assertEqual(alert_types.count("30周线上穿60周线"), 1)
                self.assertEqual(alert_types.count("有效突破60周线"), 1)

    def test_buy_alerts_skip_single_technical_signal_without_confirmation(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=95.0,
                boll_lower=92.0,
                boll_upper=110.0,
                dividend_yield=2.5,
                quant_probability=72.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 72}, {"label": "周线共振", "score": 68}], ensure_ascii=False),
                trigger_state="250日线",
                trigger_detail="测试买入提醒-单一技术信号",
                triggered_labels=("250日线",),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_buy_alerts_skip_when_quant_is_weak_or_sell_conflicts_exist(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=94.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=98.0,
                boll_lower=95.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=45.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 45}, {"label": "周线共振", "score": 42}], ensure_ascii=False),
                trigger_state="250日线、BOLL中轨、BOLL下轨、股息率",
                trigger_detail="测试买入提醒-量化偏弱",
                triggered_labels=("250日线", "BOLL中轨", "BOLL下轨", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "40"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "BOLL中轨", "BOLL下轨", "股息率", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_buy_alerts_skip_when_expected_upside_is_too_small(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=108.0,
                ma_250=110.0,
                ma_30w=115.0,
                ma_60w=100.0,
                prev_ma_30w=114.0,
                prev_ma_60w=99.0,
                boll_mid=109.0,
                boll_lower=107.0,
                boll_upper=111.0,
                dividend_yield=5.0,
                quant_probability=90.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 90}, {"label": "周线共振", "score": 86}], ensure_ascii=False),
                trigger_state="250日线、BOLL中轨、股息率",
                trigger_detail="测试买入提醒-上涨空间不足",
                triggered_labels=("250日线", "BOLL中轨", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "BOLL中轨", "股息率", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_buy_alerts_skip_when_market_environment_is_weak(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=100.0,
                boll_lower=96.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=76.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 76}, {"label": "周线共振", "score": 72}], ensure_ascii=False),
                trigger_state="250日线、BOLL中轨、股息率",
                trigger_detail="测试买入提醒-弱市过滤",
                triggered_labels=("250日线", "BOLL中轨", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
                market_environment="偏弱",
                market_bias_score=-26.0,
                industry_environment="偏弱",
                industry_bias_score=-18.0,
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "BOLL中轨", "股息率", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_buy_alerts_skip_when_volume_confirmation_is_missing(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=100.0,
                boll_lower=96.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=88.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 88}, {"label": "周线共振", "score": 84}], ensure_ascii=False),
                trigger_state="250日线、BOLL中轨、股息率",
                trigger_detail="测试买入提醒-缩量过滤",
                triggered_labels=("250日线", "BOLL中轨", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
                latest_volume_ratio=0.62,
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "BOLL中轨", "股息率", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_buy_alerts_skip_near_earnings_window_without_extra_strength(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=100.0,
                boll_lower=96.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=76.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 76}, {"label": "周线共振", "score": 73}], ensure_ascii=False),
                trigger_state="250日线、BOLL中轨、股息率",
                trigger_detail="测试买入提醒-财报窗口过滤",
                triggered_labels=("250日线", "BOLL中轨", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
                earnings_phase="财报窗口临近",
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "BOLL中轨", "股息率", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_sell_alerts_skip_shallow_upper_band_touch(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=115.4,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=108.0,
                prev_ma_30w=109.0,
                prev_ma_60w=107.0,
                boll_mid=110.0,
                boll_lower=100.0,
                boll_upper=115.0,
                dividend_yield=4.2,
                quant_probability=72.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 82}, {"label": "周线共振", "score": 76}], ensure_ascii=False),
                trigger_state="BOLL上轨卖出",
                trigger_detail="测试卖出提醒-轻微触上轨",
                triggered_labels=("BOLL上轨卖出",),
                weekly_crossed=False,
                weekly_close=115.4,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                add_trade_record(database_file.name, "admin", "600519", "buy", 100.0, 100)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:30:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] == "BOLL上轨卖出"
                ]
                self.assertEqual(len(alerts), 0)

    def test_buy_alerts_skip_when_position_is_already_too_heavy(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=99.0,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=100.0,
                prev_ma_30w=109.0,
                prev_ma_60w=99.0,
                boll_mid=100.0,
                boll_lower=96.0,
                boll_upper=110.0,
                dividend_yield=5.0,
                quant_probability=86.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 86}, {"label": "周线共振", "score": 82}], ensure_ascii=False),
                trigger_state="250日线、BOLL中轨、股息率",
                trigger_detail="测试买入提醒-重仓抑制",
                triggered_labels=("250日线", "BOLL中轨", "股息率"),
                weekly_crossed=False,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/portfolio", data={"total_investment_amount": "15000", "next_path": "/dashboard"}, follow_redirects=False)
                add_trade_record(database_file.name, "admin", "600519", "buy", 100.0, 60)
                client.post("/settings/quant", data={"enabled": "1", "probability_threshold": "50"}, follow_redirects=False)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] in {"250日线", "BOLL中轨", "股息率", "量化盈利概率"}
                ]
                self.assertEqual(len(alerts), 0)

    def test_sell_alerts_fire_earlier_for_heavy_positions(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            snapshot = SnapshotComputation(
                symbol="600519",
                display_name="贵州茅台",
                latest_price=115.4,
                ma_250=100.0,
                ma_30w=110.0,
                ma_60w=108.0,
                prev_ma_30w=109.0,
                prev_ma_60w=107.0,
                boll_mid=110.0,
                boll_lower=100.0,
                boll_upper=115.0,
                dividend_yield=4.2,
                quant_probability=72.0,
                quant_model_breakdown=json.dumps([{"label": "趋势跟随", "score": 82}, {"label": "周线共振", "score": 76}], ensure_ascii=False),
                trigger_state="BOLL上轨卖出",
                trigger_detail="测试卖出提醒-重仓提前止盈",
                triggered_labels=("BOLL上轨卖出",),
                weekly_crossed=False,
                weekly_close=115.4,
                updated_at=datetime.fromisoformat("2026-03-13T10:00:00+08:00"),
            )
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
                client.post("/settings/portfolio", data={"total_investment_amount": "20000", "next_path": "/dashboard"}, follow_redirects=False)
                add_trade_record(database_file.name, "admin", "600519", "buy", 100.0, 70)
                with patch.object(app.state.monitor, "build_snapshot", return_value=snapshot):
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:00:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-13T10:30:00+08:00"))
                alerts = [
                    row for row in get_alert_history(database_file.name, "admin", symbol="600519", days=365)
                    if row["trigger_type"] == "BOLL上轨卖出"
                ]
                self.assertEqual(len(alerts), 1)

    def test_post_close_holding_review_email_is_sent_once_per_day(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as database_file:
            app = create_app(AppSettings(db_path=database_file.name, provider_name="mock", default_symbols=()))
            with TestClient(app) as client:
                client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
                client.post("/stocks", data={"symbols_text": "600519"}, follow_redirects=False)
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
                add_trade_record(database_file.name, "admin", "600519", "buy", 100.0, 100)
                with patch("src.ai_stock_monitoring.monitor.send_message") as mocked_send:
                    mocked_send.return_value.success = True
                    mocked_send.return_value.status = "发送成功"
                    mocked_send.return_value.error = None
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-12T15:10:00+08:00"))
                    app.state.monitor.run_cycle(datetime.fromisoformat("2026-03-12T15:20:00+08:00"))
                review_calls = [call for call in mocked_send.call_args_list if "收盘持仓复盘" in call.kwargs.get("subject", "")]
                self.assertEqual(len(review_calls), 1)

    def test_alert_email_body_contains_market_context(self) -> None:
        body = build_alert_email_body(
            {
                "symbol": "600519",
                "display_name": "贵州茅台",
                "trigger_type": "250日线",
                "current_price": 1450.0,
                "detail": "测试提醒详情",
                "indicator_values": {"ma_250": 1440.0},
                "triggered_at": "2026-03-13T10:00:00+08:00",
                "market_environment": "偏弱",
                "market_bias_score": -24,
                "industry_name": "白酒",
                "industry_environment": "偏弱",
                "latest_volume_ratio": 0.78,
                "earnings_phase": "财报窗口临近",
            }
        )
        self.assertIn("250日线：1440.00", body)
        self.assertIn("现价偏离：+0.69%", body)
        self.assertIn("环境因子：大盘 偏弱(-24)", body)
        self.assertIn("行业 白酒 偏弱", body)
        self.assertIn("量能比 0.78", body)
        self.assertIn("财报节奏 财报窗口临近", body)

    def test_alert_email_html_body_contains_threshold_cards(self) -> None:
        body = build_alert_email_html_body(
            {
                "symbol": "600519",
                "display_name": "贵州茅台",
                "trigger_type": "BOLL上轨卖出",
                "current_price": 1450.0,
                "detail": "测试提醒详情",
                "indicator_values": {"boll_upper": 1430.0},
                "triggered_at": "2026-03-13T10:00:00+08:00",
                "trigger_interpretation": "这是一条局部止盈或风控提醒。",
            }
        )
        self.assertIn("周BOLL上轨", body)
        self.assertIn("+1.40%", body)
        self.assertIn("卖出侧风控", body)

    def test_portfolio_review_email_contains_report_sections(self) -> None:
        body = build_portfolio_review_email_body(
            {
                "owner_username": "admin",
                "trade_date": "2026-03-13",
                "model_learning": {
                    "overview_lines": [
                        "近 180 天纸面交易里，专业组暂时学习领先，当前状态为学习有效。",
                        "今日持仓决策里，专业组当前影响更大，贡献约 56.40% 。",
                    ],
                    "groups": [
                        {
                            "label": "专业组",
                            "learning_status": "学习有效",
                            "sample_size": 12,
                            "open_count": 2,
                            "hit_rate": 66.67,
                            "avg_return_pct": 4.28,
                            "max_drawdown_pct": 6.1,
                            "impact_degree": "高",
                            "contribution_pct": 56.4,
                            "calibration_weight": 1.14,
                            "impact_summary": "当前组内平均分 78.60，对今日最终量化结论贡献约 56.40% ，纸面交易调权系数 1.14。",
                        },
                        {
                            "label": "自适应组",
                            "learning_status": "温和有效",
                            "sample_size": 9,
                            "open_count": 3,
                            "hit_rate": 55.56,
                            "avg_return_pct": 1.92,
                            "max_drawdown_pct": 7.8,
                            "impact_degree": "中",
                            "contribution_pct": 43.6,
                            "calibration_weight": 1.03,
                            "impact_summary": "当前组内平均分 72.40，对今日最终量化结论贡献约 43.60% ，纸面交易调权系数 1.03。",
                        },
                    ],
                    "top_models": [
                        {
                            "label": "MSCI动量",
                            "learning_status": "学习有效",
                            "sample_size": 8,
                            "hit_rate": 62.5,
                            "avg_return_pct": 4.85,
                            "max_drawdown_pct": 5.2,
                        }
                    ],
                },
                "portfolio_profile": {
                    "comprehensive_advice": "建议先控仓，再分批处理强弱仓位。",
                    "holding_ratio": 62.5,
                    "recommended_holding_ratio": "40% - 60%",
                    "risk_level": "中",
                    "holding_style": "均衡配置型",
                    "overall_adjustment_suggestions": ["总仓位略高，先回到中性仓位。"],
                    "priority_reduce_positions": ["优先减仓 贵州茅台"],
                    "priority_add_positions": ["优先加仓 中国平安"],
                    "professional_advice": ["短线先看强弱切换，中线继续跟踪估值安全边际。"],
                    "risk_reasons": ["前排重仓股占比不低。"],
                    "active_positions": [
                        {
                            "symbol": "600519",
                            "display_name": "贵州茅台",
                            "latest_price": 1450.0,
                            "weight_pct": 28.0,
                            "action": "偏卖出",
                            "risk_level": "高",
                            "recommended_buy_price_range": "1380.00 - 1405.00",
                            "recommended_sell_price_range": "1475.00 - 1505.00",
                            "buy_recommendation_level": 4,
                            "sell_recommendation_level": 8,
                            "watch_price_range": "1400.00 - 1470.00",
                            "decision_summary": "当前结论偏卖出，但现价还没到理想减仓区，若反弹到 1475.00 - 1505.00 更适合分批减仓。",
                            "action_reason": "卖出信号更强（卖出 8 分 / 买入 4 分），当前最强量化模型为 趋势跟随（82.00 分）。",
                            "buy_signal_summary": "买入侧信号：股息率",
                            "sell_signal_summary": "卖出侧信号：BOLL上轨卖出、量化走弱卖出",
                            "decision_reason_lines": ["核心思路是借反弹优化卖点，而不是继续追买。"],
                            "market_environment": "偏弱",
                            "market_bias_score": -24.0,
                            "industry_name": "白酒",
                            "industry_environment": "偏弱",
                            "latest_volume_ratio": 0.78,
                            "earnings_phase": "财报窗口临近",
                            "advice_dcf_line": "DCF：内在价值 1520.00，偏差 4.83%",
                        },
                        {
                            "symbol": "601318",
                            "display_name": "中国平安",
                            "latest_price": 52.35,
                            "weight_pct": 18.0,
                            "action": "偏买入",
                            "risk_level": "中",
                            "recommended_buy_price_range": "50.00 - 51.20",
                            "recommended_sell_price_range": "56.50 - 58.00",
                            "buy_recommendation_level": 9,
                            "sell_recommendation_level": 3,
                            "watch_price_range": "50.80 - 53.50",
                            "market_environment": "中性",
                            "market_bias_score": 2.0,
                            "industry_name": "保险",
                            "industry_environment": "偏强",
                            "latest_volume_ratio": 1.18,
                            "earnings_phase": "常规窗口",
                            "advice_dcf_line": "DCF：内在价值 57.20，偏差 9.27%",
                        },
                        {
                            "symbol": "600036",
                            "display_name": "招商银行",
                            "latest_price": 41.86,
                            "weight_pct": 14.0,
                            "action": "中性",
                            "risk_level": "低",
                            "recommended_buy_price_range": "39.80 - 40.60",
                            "recommended_sell_price_range": "44.80 - 46.00",
                            "buy_recommendation_level": 6,
                            "sell_recommendation_level": 5,
                            "watch_price_range": "40.20 - 42.30",
                            "market_environment": "中性",
                            "market_bias_score": 1.0,
                            "industry_name": "银行",
                            "industry_environment": "中性",
                            "latest_volume_ratio": 0.96,
                            "earnings_phase": "常规窗口",
                            "advice_dcf_line": "DCF：内在价值 45.10，偏差 7.74%",
                        }
                    ],
                },
            }
        )
        self.assertIn("【组合总览】", body)
        self.assertIn("【明日计划】", body)
        self.assertIn("【优先减仓】", body)
        self.assertIn("【优先加仓】", body)
        self.assertIn("【高优先级股票 TOP3】", body)
        self.assertIn("【明日关注位】", body)
        self.assertIn("【风险红灯项】", body)
        self.assertIn("【模型学习成效】", body)
        self.assertIn("专业组：状态 学习有效", body)
        self.assertIn("自适应组：状态 温和有效", body)
        self.assertIn("影响程度：高", body)
        self.assertIn("命中率 66.67%", body)
        self.assertIn("【单模型领先表现】", body)
        self.assertIn("中国平安：动作 偏买入 ｜ 仓位 18.00% ｜ 买入 9/10 ｜ 卖出 3/10", body)
        self.assertIn("贵州茅台 风险等级偏高，需优先盯盘。", body)
        self.assertIn("贵州茅台：关注 1400.00 - 1470.00", body)
        self.assertIn("结论：当前结论偏卖出", body)
        self.assertIn("原因：卖出信号更强", body)
        self.assertIn("买入侧信号：股息率", body)
        self.assertIn("卖出侧信号：BOLL上轨卖出、量化走弱卖出", body)
        self.assertIn("环境：大盘 偏弱(-24) ｜ 行业 白酒 偏弱 ｜ 量能比 0.78 ｜ 财报节奏 财报窗口临近", body)

    def test_mock_dividend_yield_uses_last_year_dividend_per_share(self) -> None:
        monitor = StockMonitor(AppSettings(provider_name="mock"))
        latest_price = monitor.provider.get_quote("600519").latest_price
        seed = sum(int(char) for char in "600519")
        expected = round((0.18 + (seed % 8) * 0.06) / latest_price * 100, 2)
        self.assertEqual(monitor.provider.get_trailing_dividend_yield("600519", latest_price), expected)


if __name__ == "__main__":
    unittest.main()
