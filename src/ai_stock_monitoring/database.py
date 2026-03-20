from __future__ import annotations

"""SQLite persistence layer with per-user data isolation.

当前版本的数据库设计把“账号”和“业务数据”分开：
- `user_account` 保存登录身份与角色
- 其它 `user_*` 表都通过 `owner_username` 归属到某个账号

这样新增账号后，每个人看到的监控股票、交易流水、邮箱配置、提醒历史都会自动隔离。
"""

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
import json
import sqlite3
from typing import Any, Iterator

from .config import AppSettings
from .quant import DEFAULT_QUANT_MODELS, DEFAULT_QUANT_STRATEGY_PARAMS, normalize_strategy_params
from .security import hash_password


@contextmanager
def get_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(settings: AppSettings) -> None:
    """Create or migrate the lightweight SQLite schema."""

    with get_connection(settings.db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_account (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_email_settings (
                owner_username TEXT PRIMARY KEY,
                recipient_email TEXT,
                smtp_server TEXT,
                sender_email TEXT,
                sender_password TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_quant_settings (
                owner_username TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                probability_threshold REAL NOT NULL DEFAULT 90,
                selected_models TEXT NOT NULL,
                strategy_params TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_portfolio_settings (
                owner_username TEXT PRIMARY KEY,
                total_investment_amount REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_monitored_stock (
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (owner_username, symbol)
            );

            CREATE TABLE IF NOT EXISTS user_stock_snapshot (
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                display_name TEXT NOT NULL,
                latest_price REAL NOT NULL,
                latest_change_amount REAL NOT NULL DEFAULT 0,
                latest_change_pct REAL NOT NULL DEFAULT 0,
                ma_250 REAL NOT NULL,
                ma_30w REAL NOT NULL,
                ma_60w REAL NOT NULL,
                boll_mid REAL NOT NULL,
                boll_lower REAL NOT NULL DEFAULT 0,
                boll_upper REAL NOT NULL DEFAULT 0,
                dividend_yield REAL NOT NULL,
                quant_probability REAL NOT NULL DEFAULT 0,
                quant_model_breakdown TEXT NOT NULL DEFAULT '',
                trigger_state TEXT NOT NULL,
                trigger_detail TEXT NOT NULL,
                latest_volume_ratio REAL NOT NULL DEFAULT 1,
                market_environment TEXT NOT NULL DEFAULT '中性',
                market_bias_score REAL NOT NULL DEFAULT 0,
                industry_name TEXT NOT NULL DEFAULT '',
                industry_environment TEXT NOT NULL DEFAULT '中性',
                industry_bias_score REAL NOT NULL DEFAULT 0,
                earnings_phase TEXT NOT NULL DEFAULT '常规窗口',
                earnings_days_to_window INTEGER NOT NULL DEFAULT 999,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_username, symbol)
            );

            CREATE TABLE IF NOT EXISTS user_alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                display_name TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                current_price REAL NOT NULL,
                indicator_values TEXT NOT NULL,
                email_status TEXT NOT NULL,
                email_error TEXT,
                triggered_at TEXT NOT NULL,
                read_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_signal_state (
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                consecutive_hits INTEGER NOT NULL DEFAULT 0,
                last_condition_met INTEGER NOT NULL DEFAULT 0,
                last_event_marker TEXT,
                pending_delivery INTEGER NOT NULL DEFAULT 0,
                deliver_on TEXT,
                pending_payload TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_username, symbol, trigger_type)
            );

            CREATE TABLE IF NOT EXISTS user_job_state (
                owner_username TEXT NOT NULL,
                job_name TEXT NOT NULL,
                last_run_marker TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_username, job_name)
            );

            CREATE TABLE IF NOT EXISTS user_trade_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                traded_at TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_trade_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                analysis_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                position_summary TEXT NOT NULL,
                market_snapshot TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_model_paper_trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                display_name TEXT NOT NULL,
                model_scope TEXT NOT NULL,
                model_key TEXT NOT NULL,
                model_label TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_date TEXT NOT NULL,
                latest_price REAL NOT NULL,
                latest_date TEXT NOT NULL,
                exit_price REAL,
                exit_date TEXT,
                holding_days INTEGER NOT NULL DEFAULT 0,
                max_return_pct REAL NOT NULL DEFAULT 0,
                min_return_pct REAL NOT NULL DEFAULT 0,
                max_drawdown_pct REAL NOT NULL DEFAULT 0,
                unrealized_return_pct REAL NOT NULL DEFAULT 0,
                realized_return_pct REAL,
                entry_reason TEXT NOT NULL DEFAULT '',
                exit_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS login_guard (
                subject TEXT PRIMARY KEY,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS login_unlock_code (
                username TEXT PRIMARY KEY,
                verification_code TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consumed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_login_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                login_at TEXT NOT NULL,
                client_host TEXT NOT NULL,
                user_agent TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_user_alert_history_owner_time
                ON user_alert_history (owner_username, triggered_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_trade_record_owner_symbol_time
                ON user_trade_record (owner_username, symbol, traded_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_trade_analysis_owner_symbol_time
                ON user_trade_analysis (owner_username, symbol, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_model_paper_trade_owner_status
                ON user_model_paper_trade (owner_username, status, latest_date DESC);
            CREATE INDEX IF NOT EXISTS idx_user_model_paper_trade_owner_model
                ON user_model_paper_trade (owner_username, model_key, latest_date DESC);
            CREATE INDEX IF NOT EXISTS idx_user_signal_state_owner_delivery
                ON user_signal_state (owner_username, pending_delivery, deliver_on);
            CREATE INDEX IF NOT EXISTS idx_user_login_history_username_time
                ON user_login_history (username, login_at DESC);
            """
        )

        _ensure_schema_migrations(connection)

        now = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT INTO user_account (username, password_hash, is_admin, created_at, updated_at)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(username) DO NOTHING
            """,
            (settings.admin_username, hash_password(settings.admin_password), now, now),
        )
        _ensure_user_config_rows(connection, settings.admin_username, now)
        _migrate_legacy_data(connection, settings, now)

        if not list_monitored_stocks(connection, settings.admin_username):
            for symbol in settings.default_symbols:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO user_monitored_stock (owner_username, symbol, display_name, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (settings.admin_username, symbol, "", now),
                )


def _migrate_legacy_data(connection: sqlite3.Connection, settings: AppSettings, now: str) -> None:
    admin_username = settings.admin_username
    if _table_exists(connection, "admin_user"):
        legacy_admin = connection.execute(
            "SELECT username, password_hash FROM admin_user WHERE id = 1"
        ).fetchone()
        if legacy_admin is not None:
            connection.execute(
                """
                INSERT INTO user_account (username, password_hash, is_admin, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(username) DO NOTHING
                """,
                (legacy_admin["username"], legacy_admin["password_hash"], now, now),
            )
            admin_username = legacy_admin["username"]
            _ensure_user_config_rows(connection, admin_username, now)

    _copy_legacy_email_settings(connection, admin_username, now)
    _copy_simple_legacy_table(
        connection,
        legacy_table="monitored_stock",
        target_table="user_monitored_stock",
        owner_username=admin_username,
        columns=("symbol", "display_name", "created_at"),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="stock_snapshot",
        target_table="user_stock_snapshot",
        owner_username=admin_username,
        columns=(
            "symbol",
            "display_name",
            "latest_price",
            "ma_250",
            "ma_30w",
            "ma_60w",
            "boll_mid",
            "dividend_yield",
            "trigger_state",
            "trigger_detail",
            "updated_at",
        ),
        transform=lambda row: (
            row["symbol"],
            row["display_name"],
            row["latest_price"],
            row["ma_250"],
            row["ma_30w"],
            row["ma_60w"],
            row["boll_mid"],
            row["dividend_yield"],
            0.0,
            "",
            row["trigger_state"],
            row["trigger_detail"],
            row["updated_at"],
        ),
        target_columns=(
            "owner_username",
            "symbol",
            "display_name",
            "latest_price",
            "ma_250",
            "ma_30w",
            "ma_60w",
            "boll_mid",
            "dividend_yield",
            "quant_probability",
            "quant_model_breakdown",
            "trigger_state",
            "trigger_detail",
            "updated_at",
        ),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="alert_history",
        target_table="user_alert_history",
        owner_username=admin_username,
        columns=(
            "symbol",
            "display_name",
            "trigger_type",
            "current_price",
            "indicator_values",
            "email_status",
            "email_error",
            "triggered_at",
            "read_at",
        ),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="signal_state",
        target_table="user_signal_state",
        owner_username=admin_username,
        columns=(
            "symbol",
            "trigger_type",
            "consecutive_hits",
            "last_condition_met",
            "last_event_marker",
            "pending_delivery",
            "deliver_on",
            "pending_payload",
            "updated_at",
        ),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="job_state",
        target_table="user_job_state",
        owner_username=admin_username,
        columns=("job_name", "last_run_marker", "updated_at"),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="trade_record",
        target_table="user_trade_record",
        owner_username=admin_username,
        columns=("symbol", "side", "price", "quantity", "traded_at", "note", "created_at"),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="trade_analysis",
        target_table="user_trade_analysis",
        owner_username=admin_username,
        columns=(
            "symbol",
            "analysis_provider",
            "model_name",
            "position_summary",
            "market_snapshot",
            "analysis_json",
            "status",
            "error_message",
            "created_at",
        ),
    )
    _copy_simple_legacy_table(
        connection,
        legacy_table="model_paper_trade",
        target_table="user_model_paper_trade",
        owner_username=admin_username,
        columns=(
            "symbol",
            "display_name",
            "model_scope",
            "model_key",
            "model_label",
            "status",
            "entry_price",
            "entry_date",
            "latest_price",
            "latest_date",
            "exit_price",
            "exit_date",
            "holding_days",
            "max_return_pct",
            "min_return_pct",
            "max_drawdown_pct",
            "unrealized_return_pct",
            "realized_return_pct",
            "entry_reason",
            "exit_reason",
            "created_at",
            "updated_at",
        ),
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in columns)


def _ensure_schema_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_model_paper_trade (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            display_name TEXT NOT NULL,
            model_scope TEXT NOT NULL,
            model_key TEXT NOT NULL,
            model_label TEXT NOT NULL,
            status TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            latest_price REAL NOT NULL,
            latest_date TEXT NOT NULL,
            exit_price REAL,
            exit_date TEXT,
            holding_days INTEGER NOT NULL DEFAULT 0,
            max_return_pct REAL NOT NULL DEFAULT 0,
            min_return_pct REAL NOT NULL DEFAULT 0,
            max_drawdown_pct REAL NOT NULL DEFAULT 0,
            unrealized_return_pct REAL NOT NULL DEFAULT 0,
            realized_return_pct REAL,
            entry_reason TEXT NOT NULL DEFAULT '',
            exit_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_portfolio_settings (
            owner_username TEXT PRIMARY KEY,
            total_investment_amount REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    if _table_exists(connection, "user_quant_settings") and not _column_exists(connection, "user_quant_settings", "strategy_params"):
        connection.execute(
            "ALTER TABLE user_quant_settings ADD COLUMN strategy_params TEXT NOT NULL DEFAULT '{}'"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "boll_lower"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN boll_lower REAL NOT NULL DEFAULT 0"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "latest_change_amount"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN latest_change_amount REAL NOT NULL DEFAULT 0"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "latest_change_pct"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN latest_change_pct REAL NOT NULL DEFAULT 0"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "boll_upper"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN boll_upper REAL NOT NULL DEFAULT 0"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "latest_volume_ratio"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN latest_volume_ratio REAL NOT NULL DEFAULT 1"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "market_environment"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN market_environment TEXT NOT NULL DEFAULT '中性'"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "market_bias_score"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN market_bias_score REAL NOT NULL DEFAULT 0"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "industry_name"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN industry_name TEXT NOT NULL DEFAULT ''"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "industry_environment"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN industry_environment TEXT NOT NULL DEFAULT '中性'"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "industry_bias_score"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN industry_bias_score REAL NOT NULL DEFAULT 0"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "earnings_phase"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN earnings_phase TEXT NOT NULL DEFAULT '常规窗口'"
        )
    if _table_exists(connection, "user_stock_snapshot") and not _column_exists(connection, "user_stock_snapshot", "earnings_days_to_window"):
        connection.execute(
            "ALTER TABLE user_stock_snapshot ADD COLUMN earnings_days_to_window INTEGER NOT NULL DEFAULT 999"
        )
    if _table_exists(connection, "user_quant_settings"):
        connection.execute(
            "UPDATE user_quant_settings SET strategy_params = ? WHERE strategy_params IS NULL OR strategy_params = ''",
            (json.dumps(DEFAULT_QUANT_STRATEGY_PARAMS, ensure_ascii=False),),
        )


def _ensure_user_config_rows(connection: sqlite3.Connection, username: str, now: str) -> None:
    connection.execute(
        """
        INSERT INTO user_email_settings (owner_username, recipient_email, smtp_server, sender_email, sender_password, updated_at)
        VALUES (?, '', '', '', '', ?)
        ON CONFLICT(owner_username) DO NOTHING
        """,
        (username, now),
    )
    connection.execute(
        """
        INSERT INTO user_quant_settings (owner_username, enabled, probability_threshold, selected_models, strategy_params, updated_at)
        VALUES (?, 0, 90, ?, ?, ?)
        ON CONFLICT(owner_username) DO NOTHING
        """,
        (
            username,
            json.dumps(list(DEFAULT_QUANT_MODELS), ensure_ascii=False),
            json.dumps(DEFAULT_QUANT_STRATEGY_PARAMS, ensure_ascii=False),
            now,
        ),
    )
    connection.execute(
        """
        INSERT INTO user_portfolio_settings (owner_username, total_investment_amount, updated_at)
        VALUES (?, 0, ?)
        ON CONFLICT(owner_username) DO NOTHING
        """,
        (username, now),
    )


def _copy_legacy_email_settings(connection: sqlite3.Connection, owner_username: str, now: str) -> None:
    if not _table_exists(connection, "email_settings"):
        return
    existing = connection.execute(
        "SELECT owner_username FROM user_email_settings WHERE owner_username = ?",
        (owner_username,),
    ).fetchone()
    row = connection.execute(
        "SELECT recipient_email, smtp_server, sender_email, sender_password, updated_at FROM email_settings WHERE id = 1"
    ).fetchone()
    if row is None or existing is None:
        return
    blank = not any(existing_value for existing_value in connection.execute(
        "SELECT recipient_email, smtp_server, sender_email, sender_password FROM user_email_settings WHERE owner_username = ?",
        (owner_username,),
    ).fetchone())
    if not blank:
        return
    connection.execute(
        """
        UPDATE user_email_settings
        SET recipient_email = ?, smtp_server = ?, sender_email = ?, sender_password = ?, updated_at = ?
        WHERE owner_username = ?
        """,
        (
            row["recipient_email"] or "",
            row["smtp_server"] or "",
            row["sender_email"] or "",
            row["sender_password"] or "",
            row["updated_at"] or now,
            owner_username,
        ),
    )


def _copy_simple_legacy_table(
    connection: sqlite3.Connection,
    legacy_table: str,
    target_table: str,
    owner_username: str,
    columns: tuple[str, ...],
    transform: Any | None = None,
    target_columns: tuple[str, ...] | None = None,
) -> None:
    if not _table_exists(connection, legacy_table):
        return
    target_count = connection.execute(f"SELECT COUNT(*) AS count FROM {target_table} WHERE owner_username = ?", (owner_username,)).fetchone()["count"]
    if target_count:
        return
    select_columns = ", ".join(columns)
    rows = connection.execute(f"SELECT {select_columns} FROM {legacy_table}").fetchall()
    if not rows:
        return
    target_columns = target_columns or (("owner_username",) + columns)
    placeholder = ", ".join("?" for _ in target_columns)
    sql = f"INSERT INTO {target_table} ({', '.join(target_columns)}) VALUES ({placeholder})"
    for row in rows:
        payload = transform(row) if transform else tuple(row[column] for column in columns)
        connection.execute(sql, (owner_username, *payload))


def create_user(db_path: str, username: str, password_hash: str, is_admin: bool = False) -> None:
    now = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_account (username, password_hash, is_admin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, password_hash, 1 if is_admin else 0, now, now),
        )
        _ensure_user_config_rows(connection, username, now)


def _ensure_user_config_rows_for_db_path(db_path: str, username: str) -> None:
    now = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        _ensure_user_config_rows(connection, username, now)


def list_users(db_path: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT ua.username,
                       ua.is_admin,
                       ua.created_at,
                       ua.updated_at,
                       ll.latest_login_at
                FROM user_account ua
                LEFT JOIN (
                    SELECT username, MAX(login_at) AS latest_login_at
                    FROM user_login_history
                    GROUP BY username
                ) ll
                  ON ll.username = ua.username
                ORDER BY ua.is_admin DESC, ua.username ASC
                """
            ).fetchall()
        )


def get_user(db_path: str, username: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT username, password_hash, is_admin, created_at, updated_at FROM user_account WHERE username = ?",
            (username,),
        ).fetchone()


def get_admin_user(db_path: str) -> sqlite3.Row:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT username, password_hash, is_admin, created_at, updated_at FROM user_account WHERE is_admin = 1 ORDER BY username LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("Admin user not initialized")
    return row


def update_user_password(db_path: str, username: str, password_hash: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE user_account SET password_hash = ?, updated_at = ? WHERE username = ?",
            (password_hash, datetime.now(UTC).isoformat(), username),
        )


def update_admin_password(db_path: str, password_hash: str) -> None:
    admin = get_admin_user(db_path)
    update_user_password(db_path, admin["username"], password_hash)


def get_portfolio_settings(db_path: str, owner_username: str) -> sqlite3.Row:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT owner_username, total_investment_amount, updated_at FROM user_portfolio_settings WHERE owner_username = ?",
            (owner_username,),
        ).fetchone()
    if row is None:
        _ensure_user_config_rows_for_db_path(db_path, owner_username)
        with get_connection(db_path) as connection:
            row = connection.execute(
                "SELECT owner_username, total_investment_amount, updated_at FROM user_portfolio_settings WHERE owner_username = ?",
                (owner_username,),
            ).fetchone()
    if row is None:
        raise RuntimeError("Portfolio settings not initialized")
    return row


def save_portfolio_settings(db_path: str, owner_username: str, total_investment_amount: float) -> None:
    updated_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE user_portfolio_settings
            SET total_investment_amount = ?, updated_at = ?
            WHERE owner_username = ?
            """,
            (max(0.0, float(total_investment_amount)), updated_at, owner_username),
        )


def get_email_settings(db_path: str, owner_username: str) -> sqlite3.Row:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT owner_username, recipient_email, smtp_server, sender_email, sender_password, updated_at FROM user_email_settings WHERE owner_username = ?",
            (owner_username,),
        ).fetchone()
    if row is None:
        _ensure_user_config_rows_for_db_path(db_path, owner_username)
        with get_connection(db_path) as connection:
            row = connection.execute(
                "SELECT owner_username, recipient_email, smtp_server, sender_email, sender_password, updated_at FROM user_email_settings WHERE owner_username = ?",
                (owner_username,),
            ).fetchone()
    if row is None:
        raise RuntimeError("Email settings not initialized")
    return row


def save_email_settings(
    db_path: str,
    owner_username: str,
    recipient_email: str,
    smtp_server: str,
    sender_email: str,
    sender_password: str,
) -> None:
    now = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE user_email_settings
            SET recipient_email = ?, smtp_server = ?, sender_email = ?, sender_password = ?, updated_at = ?
            WHERE owner_username = ?
            """,
            (recipient_email, smtp_server, sender_email, sender_password, now, owner_username),
        )


def get_quant_settings(db_path: str, owner_username: str) -> sqlite3.Row:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT owner_username, enabled, probability_threshold, selected_models, strategy_params, updated_at FROM user_quant_settings WHERE owner_username = ?",
            (owner_username,),
        ).fetchone()
    if row is None:
        _ensure_user_config_rows_for_db_path(db_path, owner_username)
        with get_connection(db_path) as connection:
            row = connection.execute(
                "SELECT owner_username, enabled, probability_threshold, selected_models, strategy_params, updated_at FROM user_quant_settings WHERE owner_username = ?",
                (owner_username,),
            ).fetchone()
    if row is None:
        raise RuntimeError("Quant settings not initialized")
    return row


def save_quant_settings(
    db_path: str,
    owner_username: str,
    enabled: bool,
    probability_threshold: float,
    selected_models: list[str],
    strategy_params: dict[str, float | bool] | None = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    normalized_params = normalize_strategy_params(strategy_params)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE user_quant_settings
            SET enabled = ?, probability_threshold = ?, selected_models = ?, strategy_params = ?, updated_at = ?
            WHERE owner_username = ?
            """,
            (
                1 if enabled else 0,
                probability_threshold,
                json.dumps(selected_models, ensure_ascii=False),
                json.dumps(normalized_params, ensure_ascii=False),
                now,
                owner_username,
            ),
        )


def get_login_guard_state(db_path: str, subject: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT subject, failed_attempts, locked_until, updated_at FROM login_guard WHERE subject = ?",
            (subject,),
        ).fetchone()


def record_failed_login(db_path: str, subject: str, lock_minutes: int = 15) -> sqlite3.Row:
    current = get_login_guard_state(db_path, subject)
    failed_attempts = 1 if current is None else int(current["failed_attempts"]) + 1
    locked_until = None
    if failed_attempts >= 5:
        locked_until = (datetime.now(UTC) + timedelta(minutes=lock_minutes)).isoformat()
    updated_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO login_guard (subject, failed_attempts, locked_until, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(subject) DO UPDATE SET
                failed_attempts = excluded.failed_attempts,
                locked_until = excluded.locked_until,
                updated_at = excluded.updated_at
            """,
            (subject, failed_attempts, locked_until, updated_at),
        )
    guard = get_login_guard_state(db_path, subject)
    if guard is None:
        raise RuntimeError("Login guard state not persisted")
    return guard


def clear_login_guard_state(db_path: str, subject: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM login_guard WHERE subject = ?", (subject,))


def clear_login_guard_states_for_username(db_path: str, username: str) -> None:
    """Clear all temporary lock records associated with one account.

    登录锁定是按 `username|ip|user-agent` 这类 subject 维度存储的。
    邮箱确认解封时，需要把该账号对应的所有 subject 一并清掉。
    """

    with get_connection(db_path) as connection:
        connection.execute(
            "DELETE FROM login_guard WHERE subject LIKE ?",
            (f"{username.lower()}|%",),
        )


def save_login_unlock_code(
    db_path: str,
    username: str,
    verification_code: str,
    expires_at: str,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO login_unlock_code (username, verification_code, expires_at, created_at, consumed_at)
            VALUES (?, ?, ?, ?, NULL)
            ON CONFLICT(username) DO UPDATE SET
                verification_code = excluded.verification_code,
                expires_at = excluded.expires_at,
                created_at = excluded.created_at,
                consumed_at = NULL
            """,
            (username, verification_code, expires_at, created_at),
        )


def get_login_unlock_code(db_path: str, username: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            """
            SELECT username, verification_code, expires_at, created_at, consumed_at
            FROM login_unlock_code
            WHERE username = ?
            """,
            (username,),
        ).fetchone()


def consume_login_unlock_code(db_path: str, username: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE login_unlock_code SET consumed_at = ? WHERE username = ?",
            (datetime.now(UTC).isoformat(), username),
        )


def record_login_event(db_path: str, username: str, client_host: str, user_agent: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_login_history (username, login_at, client_host, user_agent)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                datetime.now(UTC).isoformat(),
                (client_host or "unknown")[:128],
                (user_agent or "unknown")[:255],
            ),
        )


def list_recent_login_events(db_path: str, username: str, limit: int = 5) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, username, login_at, client_host, user_agent
                FROM user_login_history
                WHERE username = ?
                ORDER BY login_at DESC
                LIMIT ?
                """,
                (username, limit),
            ).fetchall()
        )


def list_monitored_stocks(connection: sqlite3.Connection, owner_username: str | None = None) -> list[sqlite3.Row]:
    query = "SELECT owner_username, symbol, display_name, created_at FROM user_monitored_stock"
    parameters: list[str] = []
    if owner_username:
        query += " WHERE owner_username = ?"
        parameters.append(owner_username)
    query += " ORDER BY owner_username, symbol"
    return list(connection.execute(query, parameters).fetchall())


def get_monitored_stocks(db_path: str, owner_username: str | None = None) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list_monitored_stocks(connection, owner_username)


def add_monitored_stock(db_path: str, owner_username: str, symbol: str, display_name: str = "") -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO user_monitored_stock (owner_username, symbol, display_name, created_at) VALUES (?, ?, ?, ?)",
            (owner_username, symbol, display_name, datetime.now(UTC).isoformat()),
        )


def remove_monitored_stock(db_path: str, owner_username: str, symbol: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM user_monitored_stock WHERE owner_username = ? AND symbol = ?", (owner_username, symbol))
        connection.execute("DELETE FROM user_stock_snapshot WHERE owner_username = ? AND symbol = ?", (owner_username, symbol))
        connection.execute("DELETE FROM user_signal_state WHERE owner_username = ? AND symbol = ?", (owner_username, symbol))


def get_snapshots(db_path: str, owner_username: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT ms.owner_username,
                       ms.symbol,
                       COALESCE(ss.display_name, ms.display_name, '') AS display_name,
                       ss.latest_price,
                       ss.latest_change_amount,
                       ss.latest_change_pct,
                       ss.ma_250,
                       ss.ma_30w,
                       ss.ma_60w,
                       ss.boll_mid,
                       ss.boll_lower,
                       ss.boll_upper,
                       ss.dividend_yield,
                       ss.quant_probability,
                       ss.quant_model_breakdown,
                       ss.trigger_state,
                       ss.trigger_detail,
                       ss.latest_volume_ratio,
                       ss.market_environment,
                       ss.market_bias_score,
                       ss.industry_name,
                       ss.industry_environment,
                       ss.industry_bias_score,
                       ss.earnings_phase,
                       ss.earnings_days_to_window,
                       ss.updated_at
                FROM user_monitored_stock ms
                LEFT JOIN user_stock_snapshot ss
                  ON ms.owner_username = ss.owner_username AND ms.symbol = ss.symbol
                WHERE ms.owner_username = ?
                ORDER BY ms.symbol
                """,
                (owner_username,),
            ).fetchall()
        )


def get_snapshot(db_path: str, owner_username: str, symbol: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM user_stock_snapshot WHERE owner_username = ? AND symbol = ?",
            (owner_username, symbol),
        ).fetchone()


def upsert_snapshot(
    db_path: str,
    owner_username: str,
    symbol: str,
    display_name: str,
    latest_price: float,
    latest_change_amount: float,
    latest_change_pct: float,
    ma_250: float,
    ma_30w: float,
    ma_60w: float,
    boll_mid: float,
    boll_lower: float,
    boll_upper: float,
    dividend_yield: float,
    quant_probability: float,
    quant_model_breakdown: str,
    trigger_state: str,
    trigger_detail: str,
    latest_volume_ratio: float = 1.0,
    market_environment: str = "中性",
    market_bias_score: float = 0.0,
    industry_name: str = "",
    industry_environment: str = "中性",
    industry_bias_score: float = 0.0,
    earnings_phase: str = "常规窗口",
    earnings_days_to_window: int = 999,
    updated_at: str = "",
) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_stock_snapshot (
                owner_username, symbol, display_name, latest_price, latest_change_amount, latest_change_pct, ma_250, ma_30w, ma_60w,
                boll_mid, boll_lower, boll_upper, dividend_yield, quant_probability, quant_model_breakdown,
                trigger_state, trigger_detail, latest_volume_ratio, market_environment, market_bias_score,
                industry_name, industry_environment, industry_bias_score, earnings_phase, earnings_days_to_window, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_username, symbol) DO UPDATE SET
                display_name = excluded.display_name,
                latest_price = excluded.latest_price,
                latest_change_amount = excluded.latest_change_amount,
                latest_change_pct = excluded.latest_change_pct,
                ma_250 = excluded.ma_250,
                ma_30w = excluded.ma_30w,
                ma_60w = excluded.ma_60w,
                boll_mid = excluded.boll_mid,
                boll_lower = excluded.boll_lower,
                boll_upper = excluded.boll_upper,
                dividend_yield = excluded.dividend_yield,
                quant_probability = excluded.quant_probability,
                quant_model_breakdown = excluded.quant_model_breakdown,
                trigger_state = excluded.trigger_state,
                trigger_detail = excluded.trigger_detail,
                latest_volume_ratio = excluded.latest_volume_ratio,
                market_environment = excluded.market_environment,
                market_bias_score = excluded.market_bias_score,
                industry_name = excluded.industry_name,
                industry_environment = excluded.industry_environment,
                industry_bias_score = excluded.industry_bias_score,
                earnings_phase = excluded.earnings_phase,
                earnings_days_to_window = excluded.earnings_days_to_window,
                updated_at = excluded.updated_at
            """,
            (
                owner_username,
                symbol,
                display_name,
                latest_price,
                latest_change_amount,
                latest_change_pct,
                ma_250,
                ma_30w,
                ma_60w,
                boll_mid,
                boll_lower,
                boll_upper,
                dividend_yield,
                quant_probability,
                quant_model_breakdown,
                trigger_state,
                trigger_detail,
                latest_volume_ratio,
                market_environment,
                market_bias_score,
                industry_name,
                industry_environment,
                industry_bias_score,
                earnings_phase,
                earnings_days_to_window,
                updated_at,
            ),
        )


def add_alert_history(
    db_path: str,
    owner_username: str,
    symbol: str,
    display_name: str,
    trigger_type: str,
    current_price: float,
    indicator_values: dict[str, float | str],
    email_status: str,
    email_error: str | None,
    triggered_at: str,
) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_alert_history (
                owner_username, symbol, display_name, trigger_type, current_price, indicator_values,
                email_status, email_error, triggered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_username,
                symbol,
                display_name,
                trigger_type,
                current_price,
                json.dumps(indicator_values, ensure_ascii=False),
                email_status,
                email_error,
                triggered_at,
            ),
        )


def get_alert_history(
    db_path: str,
    owner_username: str,
    symbol: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    days: int = 90,
) -> list[sqlite3.Row]:
    start = date_from or (datetime.now(UTC).date() - timedelta(days=days))
    end = date_to or datetime.now(UTC).date()
    query = """
        SELECT id, owner_username, symbol, display_name, trigger_type, current_price, indicator_values,
               email_status, email_error, triggered_at, read_at
        FROM user_alert_history
        WHERE owner_username = ?
          AND date(triggered_at) BETWEEN ? AND ?
    """
    parameters: list[str] = [owner_username, start.isoformat(), end.isoformat()]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY triggered_at DESC"
    with get_connection(db_path) as connection:
        return list(connection.execute(query, parameters).fetchall())


def get_unread_alerts(db_path: str, owner_username: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, owner_username, symbol, display_name, trigger_type, current_price, indicator_values,
                       email_status, triggered_at
                FROM user_alert_history
                WHERE owner_username = ? AND read_at IS NULL
                ORDER BY triggered_at DESC
                LIMIT 5
                """,
                (owner_username,),
            ).fetchall()
        )


def mark_alert_as_read(db_path: str, owner_username: str, alert_id: int) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE user_alert_history SET read_at = ? WHERE owner_username = ? AND id = ?",
            (datetime.now(UTC).isoformat(), owner_username, alert_id),
        )


def get_signal_state(db_path: str, owner_username: str, symbol: str, trigger_type: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM user_signal_state WHERE owner_username = ? AND symbol = ? AND trigger_type = ?",
            (owner_username, symbol, trigger_type),
        ).fetchone()


def upsert_signal_state(
    db_path: str,
    owner_username: str,
    symbol: str,
    trigger_type: str,
    consecutive_hits: int,
    last_condition_met: bool,
    last_event_marker: str | None,
    pending_delivery: bool,
    deliver_on: str | None,
    pending_payload: dict[str, Any] | None,
) -> None:
    updated_at = datetime.now(UTC).isoformat()
    payload_text = json.dumps(pending_payload, ensure_ascii=False) if pending_payload else None
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_signal_state (
                owner_username, symbol, trigger_type, consecutive_hits, last_condition_met,
                last_event_marker, pending_delivery, deliver_on, pending_payload, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_username, symbol, trigger_type) DO UPDATE SET
                consecutive_hits = excluded.consecutive_hits,
                last_condition_met = excluded.last_condition_met,
                last_event_marker = excluded.last_event_marker,
                pending_delivery = excluded.pending_delivery,
                deliver_on = excluded.deliver_on,
                pending_payload = excluded.pending_payload,
                updated_at = excluded.updated_at
            """,
            (
                owner_username,
                symbol,
                trigger_type,
                consecutive_hits,
                1 if last_condition_met else 0,
                last_event_marker,
                1 if pending_delivery else 0,
                deliver_on,
                payload_text,
                updated_at,
            ),
        )


def list_pending_signal_states(db_path: str, deliver_on_or_before: str, owner_username: str | None = None) -> list[sqlite3.Row]:
    query = """
        SELECT *
        FROM user_signal_state
        WHERE pending_delivery = 1
          AND deliver_on IS NOT NULL
          AND deliver_on <= ?
    """
    parameters: list[str] = [deliver_on_or_before]
    if owner_username:
        query += " AND owner_username = ?"
        parameters.append(owner_username)
    query += " ORDER BY deliver_on, owner_username, symbol"
    with get_connection(db_path) as connection:
        return list(connection.execute(query, parameters).fetchall())


def get_job_state(db_path: str, owner_username: str, job_name: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM user_job_state WHERE owner_username = ? AND job_name = ?",
            (owner_username, job_name),
        ).fetchone()


def set_job_state(db_path: str, owner_username: str, job_name: str, marker: str) -> None:
    updated_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_job_state (owner_username, job_name, last_run_marker, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(owner_username, job_name) DO UPDATE SET
                last_run_marker = excluded.last_run_marker,
                updated_at = excluded.updated_at
            """,
            (owner_username, job_name, marker, updated_at),
        )


def add_trade_record(
    db_path: str,
    owner_username: str,
    symbol: str,
    side: str,
    price: float,
    quantity: int,
    traded_at: str | None = None,
    note: str = "",
) -> None:
    created_at = datetime.now(UTC).isoformat()
    normalized_price = round(float(price), 5)
    normalized_traded_at = traded_at or datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_trade_record (owner_username, symbol, side, price, quantity, traded_at, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (owner_username, symbol, side, normalized_price, quantity, normalized_traded_at, note, created_at),
        )


def count_trade_records(db_path: str, owner_username: str, symbol: str | None = None) -> int:
    query = "SELECT COUNT(1) AS total FROM user_trade_record WHERE owner_username = ?"
    parameters: list[str] = [owner_username]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    with get_connection(db_path) as connection:
        row = connection.execute(query, parameters).fetchone()
    return int(row["total"] if row else 0)


def list_trade_records(db_path: str, owner_username: str, symbol: str | None = None) -> list[sqlite3.Row]:
    query = """
        SELECT id, owner_username, symbol, side, price, quantity, traded_at, note, created_at
        FROM user_trade_record
        WHERE owner_username = ?
    """
    parameters: list[str] = [owner_username]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY datetime(traded_at) DESC, id DESC"
    with get_connection(db_path) as connection:
        return list(connection.execute(query, parameters).fetchall())


def list_trade_records_paginated(
    db_path: str,
    owner_username: str,
    symbol: str | None = None,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[sqlite3.Row], int]:
    normalized_page = max(1, int(page))
    normalized_page_size = max(1, min(100, int(page_size)))
    total = count_trade_records(db_path, owner_username, symbol)
    query = """
        SELECT id, owner_username, symbol, side, price, quantity, traded_at, note, created_at
        FROM user_trade_record
        WHERE owner_username = ?
    """
    parameters: list[Any] = [owner_username]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY datetime(traded_at) DESC, id DESC LIMIT ? OFFSET ?"
    parameters.extend([normalized_page_size, (normalized_page - 1) * normalized_page_size])
    with get_connection(db_path) as connection:
        rows = list(connection.execute(query, parameters).fetchall())
    return rows, total


def list_trade_records_for_symbol(db_path: str, owner_username: str, symbol: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, owner_username, symbol, side, price, quantity, traded_at, note, created_at
                FROM user_trade_record
                WHERE owner_username = ? AND symbol = ?
                ORDER BY datetime(traded_at) ASC, id ASC
                """,
                (owner_username, symbol),
            ).fetchall()
        )


def delete_trade_record(db_path: str, owner_username: str, trade_id: int) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "DELETE FROM user_trade_record WHERE owner_username = ? AND id = ?",
            (owner_username, trade_id),
        )


def delete_trade_records_for_symbol(db_path: str, owner_username: str, symbol: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "DELETE FROM user_trade_record WHERE owner_username = ? AND symbol = ?",
            (owner_username, symbol),
        )
        connection.execute(
            "DELETE FROM user_trade_analysis WHERE owner_username = ? AND symbol = ?",
            (owner_username, symbol),
        )


def add_trade_analysis(
    db_path: str,
    owner_username: str,
    symbol: str,
    analysis_provider: str,
    model_name: str,
    position_summary: dict[str, Any],
    market_snapshot: dict[str, Any],
    analysis_json: dict[str, Any],
    status: str,
    error_message: str | None,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_trade_analysis (
                owner_username, symbol, analysis_provider, model_name, position_summary, market_snapshot,
                analysis_json, status, error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_username,
                symbol,
                analysis_provider,
                model_name,
                json.dumps(position_summary, ensure_ascii=False),
                json.dumps(market_snapshot, ensure_ascii=False),
                json.dumps(analysis_json, ensure_ascii=False),
                status,
                error_message,
                created_at,
            ),
        )


def get_latest_trade_analysis(db_path: str, owner_username: str, symbol: str | None = None) -> sqlite3.Row | None:
    query = """
        SELECT id, owner_username, symbol, analysis_provider, model_name, position_summary, market_snapshot,
               analysis_json, status, error_message, created_at
        FROM user_trade_analysis
        WHERE owner_username = ?
    """
    parameters: list[str] = [owner_username]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY created_at DESC, id DESC LIMIT 1"
    with get_connection(db_path) as connection:
        return connection.execute(query, parameters).fetchone()


def get_open_model_paper_trade(
    db_path: str,
    owner_username: str,
    symbol: str,
    model_key: str,
) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            """
            SELECT *
            FROM user_model_paper_trade
            WHERE owner_username = ? AND symbol = ? AND model_key = ? AND status = 'open'
            ORDER BY id DESC
            LIMIT 1
            """,
            (owner_username, symbol, model_key),
        ).fetchone()


def list_model_paper_trades(
    db_path: str,
    owner_username: str,
    symbol: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[sqlite3.Row]:
    query = """
        SELECT *
        FROM user_model_paper_trade
        WHERE owner_username = ?
    """
    parameters: list[Any] = [owner_username]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    if status:
        query += " AND status = ?"
        parameters.append(status)
    if date_from:
        query += " AND COALESCE(exit_date, latest_date, entry_date) >= ?"
        parameters.append(date_from)
    if date_to:
        query += " AND entry_date <= ?"
        parameters.append(date_to)
    query += " ORDER BY latest_date DESC, id DESC"
    with get_connection(db_path) as connection:
        return list(connection.execute(query, parameters).fetchall())


def open_model_paper_trade(
    db_path: str,
    owner_username: str,
    symbol: str,
    display_name: str,
    model_scope: str,
    model_key: str,
    model_label: str,
    entry_price: float,
    entry_date: str,
    entry_reason: str,
) -> None:
    now = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_model_paper_trade (
                owner_username, symbol, display_name, model_scope, model_key, model_label,
                status, entry_price, entry_date, latest_price, latest_date,
                holding_days, max_return_pct, min_return_pct, max_drawdown_pct,
                unrealized_return_pct, entry_reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, 0, 0, 0, 0, 0, ?, ?, ?)
            """,
            (
                owner_username,
                symbol,
                display_name,
                model_scope,
                model_key,
                model_label,
                entry_price,
                entry_date,
                entry_price,
                entry_date,
                entry_reason,
                now,
                now,
            ),
        )


def mark_model_paper_trade(
    db_path: str,
    trade_id: int,
    latest_price: float,
    latest_date: str,
    holding_days: int,
    max_return_pct: float,
    min_return_pct: float,
    max_drawdown_pct: float,
    unrealized_return_pct: float,
) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE user_model_paper_trade
            SET latest_price = ?, latest_date = ?, holding_days = ?,
                max_return_pct = ?, min_return_pct = ?, max_drawdown_pct = ?,
                unrealized_return_pct = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                latest_price,
                latest_date,
                holding_days,
                max_return_pct,
                min_return_pct,
                max_drawdown_pct,
                unrealized_return_pct,
                datetime.now(UTC).isoformat(),
                trade_id,
            ),
        )


def close_model_paper_trade(
    db_path: str,
    trade_id: int,
    exit_price: float,
    exit_date: str,
    holding_days: int,
    max_return_pct: float,
    min_return_pct: float,
    max_drawdown_pct: float,
    realized_return_pct: float,
    exit_reason: str,
) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE user_model_paper_trade
            SET status = 'closed',
                latest_price = ?,
                latest_date = ?,
                exit_price = ?,
                exit_date = ?,
                holding_days = ?,
                max_return_pct = ?,
                min_return_pct = ?,
                max_drawdown_pct = ?,
                unrealized_return_pct = 0,
                realized_return_pct = ?,
                exit_reason = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                exit_price,
                exit_date,
                exit_price,
                exit_date,
                holding_days,
                max_return_pct,
                min_return_pct,
                max_drawdown_pct,
                realized_return_pct,
                exit_reason,
                datetime.now(UTC).isoformat(),
                trade_id,
            ),
        )
