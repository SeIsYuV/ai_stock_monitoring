from src.ai_stock_monitoring.monitor import StockMonitor


def main() -> None:
    monitor = StockMonitor()
    summary = monitor.run()
    print(summary)


if __name__ == "__main__":
    main()

