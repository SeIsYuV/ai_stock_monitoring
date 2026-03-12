from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    watchlist: tuple[str, ...]


def load_config() -> AppConfig:
    raw_watchlist = os.getenv("WATCHLIST", "000001.SZ,600519.SH,AAPL")
    watchlist = tuple(item.strip() for item in raw_watchlist.split(",") if item.strip())
    return AppConfig(watchlist=watchlist)

