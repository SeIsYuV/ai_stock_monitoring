from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppSettings:
    """Application configuration loaded from environment variables."""

    app_name: str = "AI Stock Monitoring"
    host: str = "127.0.0.1"
    port: int = 11223
    db_path: str = "stock_monitor.db"
    refresh_interval_seconds: int = 30
    admin_username: str = "admin"
    admin_password: str = "admin123"
    session_cookie_name: str = "stock_monitor_session"
    provider_name: str = "akshare"
    timezone_name: str = "Asia/Shanghai"
    default_symbols: tuple[str, ...] = ("000001", "600519", "300750")
    detail_chart_days: int = 60
    llm_provider_name: str = "openai"
    llm_model_name: str = "gpt-4.1-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1/responses"


def load_settings() -> AppSettings:
    """Build settings once so the app can run from shell or Docker."""

    raw_symbols = os.getenv("ASM_DEFAULT_SYMBOLS", "000001,600519,300750")
    default_symbols = tuple(
        item.strip() for item in raw_symbols.split(",") if item.strip()
    )

    return AppSettings(
        host=os.getenv("ASM_HOST", "127.0.0.1"),
        port=int(os.getenv("ASM_PORT", "11223")),
        db_path=os.getenv("ASM_DB_PATH", "stock_monitor.db"),
        refresh_interval_seconds=int(os.getenv("ASM_REFRESH_INTERVAL", "30")),
        admin_username=os.getenv("ASM_ADMIN_USERNAME", "admin"),
        admin_password=os.getenv("ASM_ADMIN_PASSWORD", "admin123"),
        provider_name=os.getenv("ASM_PROVIDER", "akshare"),
        default_symbols=default_symbols,
        detail_chart_days=int(os.getenv("ASM_DETAIL_CHART_DAYS", "60")),
        llm_provider_name=os.getenv("ASM_LLM_PROVIDER", "openai"),
        llm_model_name=os.getenv("ASM_LLM_MODEL", "gpt-4.1-mini"),
        llm_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_base_url=os.getenv("ASM_LLM_BASE_URL", "https://api.openai.com/v1/responses"),
    )
