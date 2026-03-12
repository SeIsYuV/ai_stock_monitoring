from .config import load_config


class StockMonitor:
    def __init__(self) -> None:
        self.config = load_config()

    def run(self) -> str:
        symbols = ", ".join(self.config.watchlist)
        return f"AI stock monitoring project initialized. Watching: {symbols}"

