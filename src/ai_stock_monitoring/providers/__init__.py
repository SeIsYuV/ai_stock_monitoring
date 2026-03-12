from __future__ import annotations

from .akshare_provider import AkshareMarketDataProvider
from .base import MarketDataProvider
from .mock import MockMarketDataProvider


def load_provider(provider_name: str) -> MarketDataProvider:
    if provider_name == "mock":
        return MockMarketDataProvider()
    if provider_name == "akshare":
        return AkshareMarketDataProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")
