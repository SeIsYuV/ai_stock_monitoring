from src.ai_stock_monitoring.monitor import StockMonitor


def test_monitor_run_contains_default_symbols() -> None:
    result = StockMonitor().run()
    assert "AI stock monitoring project initialized" in result
    assert "000001.SZ" in result

