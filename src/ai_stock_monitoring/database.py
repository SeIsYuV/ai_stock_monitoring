from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
import json
import sqlite3
from typing import Any, Iterator

from .config import AppSettings
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
            CREATE TABLE IF NOT EXISTS admin_user (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                username TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS monitored_stock (
                symbol TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS email_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                recipient_email TEXT,
                smtp_server TEXT,
                sender_email TEXT,
                sender_password TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stock_snapshot (
                symbol TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                latest_price REAL NOT NULL,
                ma_250 REAL NOT NULL,
                ma_30w REAL NOT NULL,
                ma_60w REAL NOT NULL,
                boll_mid REAL NOT NULL,
                dividend_yield REAL NOT NULL,
                trigger_state TEXT NOT NULL,
                trigger_detail TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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

            CREATE TABLE IF NOT EXISTS signal_state (
                symbol TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                consecutive_hits INTEGER NOT NULL DEFAULT 0,
                last_condition_met INTEGER NOT NULL DEFAULT 0,
                last_event_marker TEXT,
                pending_delivery INTEGER NOT NULL DEFAULT 0,
                deliver_on TEXT,
                pending_payload TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (symbol, trigger_type)
            );

            CREATE TABLE IF NOT EXISTS job_state (
                job_name TEXT PRIMARY KEY,
                last_run_marker TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                traded_at TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """
        )

        now = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT INTO admin_user (id, username, password_hash, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                username = excluded.username,
                password_hash = excluded.password_hash,
                updated_at = excluded.updated_at
            """,
            (settings.admin_username, hash_password(settings.admin_password), now),
        )
        connection.execute(
            """
            INSERT INTO email_settings (id, recipient_email, smtp_server, sender_email, sender_password, updated_at)
            VALUES (1, '', '', '', '', ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (now,),
        )

        if not list_monitored_stocks(connection):
            for symbol in settings.default_symbols:
                connection.execute(
                    "INSERT OR IGNORE INTO monitored_stock (symbol, display_name, created_at) VALUES (?, ?, ?)",
                    (symbol, "", now),
                )


def get_admin_user(db_path: str) -> sqlite3.Row:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT username, password_hash FROM admin_user WHERE id = 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("Admin user not initialized")
    return row


def list_monitored_stocks(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            "SELECT symbol, display_name, created_at FROM monitored_stock ORDER BY symbol"
        ).fetchall()
    )


def get_monitored_stocks(db_path: str) -> list[sqlite3.Row]:
    """Return the configured watchlist sorted by symbol."""

    with get_connection(db_path) as connection:
        return list_monitored_stocks(connection)


def add_monitored_stock(db_path: str, symbol: str, display_name: str = "") -> None:
    now = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO monitored_stock (symbol, display_name, created_at) VALUES (?, ?, ?)",
            (symbol, display_name, now),
        )


def remove_monitored_stock(db_path: str, symbol: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM monitored_stock WHERE symbol = ?", (symbol,))
        connection.execute("DELETE FROM stock_snapshot WHERE symbol = ?", (symbol,))
        connection.execute("DELETE FROM signal_state WHERE symbol = ?", (symbol,))


def get_email_settings(db_path: str) -> sqlite3.Row:
    """Load the single SMTP configuration row."""

    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT recipient_email, smtp_server, sender_email, sender_password, updated_at FROM email_settings WHERE id = 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("Email settings not initialized")
    return row


def save_email_settings(
    db_path: str,
    recipient_email: str,
    smtp_server: str,
    sender_email: str,
    sender_password: str,
) -> None:
    now = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE email_settings
            SET recipient_email = ?, smtp_server = ?, sender_email = ?, sender_password = ?, updated_at = ?
            WHERE id = 1
            """,
            (recipient_email, smtp_server, sender_email, sender_password, now),
        )


def get_snapshots(db_path: str) -> list[sqlite3.Row]:
    """Load the latest per-symbol monitoring snapshot for the dashboard."""

    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT ms.symbol,
                       COALESCE(ss.display_name, ms.display_name, '') AS display_name,
                       ss.latest_price,
                       ss.ma_250,
                       ss.ma_30w,
                       ss.ma_60w,
                       ss.boll_mid,
                       ss.dividend_yield,
                       ss.trigger_state,
                       ss.trigger_detail,
                       ss.updated_at
                FROM monitored_stock ms
                LEFT JOIN stock_snapshot ss ON ms.symbol = ss.symbol
                ORDER BY ms.symbol
                """
            ).fetchall()
        )


def get_snapshot(db_path: str, symbol: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM stock_snapshot WHERE symbol = ?", (symbol,)
        ).fetchone()


def upsert_snapshot(
    db_path: str,
    symbol: str,
    display_name: str,
    latest_price: float,
    ma_250: float,
    ma_30w: float,
    ma_60w: float,
    boll_mid: float,
    dividend_yield: float,
    trigger_state: str,
    trigger_detail: str,
    updated_at: str,
) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO stock_snapshot (
                symbol, display_name, latest_price, ma_250, ma_30w, ma_60w, boll_mid,
                dividend_yield, trigger_state, trigger_detail, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                display_name = excluded.display_name,
                latest_price = excluded.latest_price,
                ma_250 = excluded.ma_250,
                ma_30w = excluded.ma_30w,
                ma_60w = excluded.ma_60w,
                boll_mid = excluded.boll_mid,
                dividend_yield = excluded.dividend_yield,
                trigger_state = excluded.trigger_state,
                trigger_detail = excluded.trigger_detail,
                updated_at = excluded.updated_at
            """,
            (
                symbol,
                display_name,
                latest_price,
                ma_250,
                ma_30w,
                ma_60w,
                boll_mid,
                dividend_yield,
                trigger_state,
                trigger_detail,
                updated_at,
            ),
        )


def add_alert_history(
    db_path: str,
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
            INSERT INTO alert_history (
                symbol, display_name, trigger_type, current_price, indicator_values,
                email_status, email_error, triggered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
    symbol: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    days: int = 90,
) -> list[sqlite3.Row]:
    """List recent alerts with optional symbol and date filters."""

    start = date_from or (datetime.now(UTC).date() - timedelta(days=days))
    end = date_to or datetime.now(UTC).date()
    query = """
        SELECT id, symbol, display_name, trigger_type, current_price, indicator_values,
               email_status, email_error, triggered_at, read_at
        FROM alert_history
        WHERE date(triggered_at) BETWEEN ? AND ?
    """
    parameters: list[str] = [start.isoformat(), end.isoformat()]
    if symbol:
        query += " AND symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY triggered_at DESC"
    with get_connection(db_path) as connection:
        return list(connection.execute(query, parameters).fetchall())


def get_unread_alerts(db_path: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, symbol, display_name, trigger_type, current_price, indicator_values,
                       email_status, triggered_at
                FROM alert_history
                WHERE read_at IS NULL
                ORDER BY triggered_at DESC
                LIMIT 5
                """
            ).fetchall()
        )


def mark_alert_as_read(db_path: str, alert_id: int) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE alert_history SET read_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), alert_id),
        )


def get_signal_state(db_path: str, symbol: str, trigger_type: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM signal_state WHERE symbol = ? AND trigger_type = ?",
            (symbol, trigger_type),
        ).fetchone()


def upsert_signal_state(
    db_path: str,
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
            INSERT INTO signal_state (
                symbol, trigger_type, consecutive_hits, last_condition_met, last_event_marker,
                pending_delivery, deliver_on, pending_payload, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, trigger_type) DO UPDATE SET
                consecutive_hits = excluded.consecutive_hits,
                last_condition_met = excluded.last_condition_met,
                last_event_marker = excluded.last_event_marker,
                pending_delivery = excluded.pending_delivery,
                deliver_on = excluded.deliver_on,
                pending_payload = excluded.pending_payload,
                updated_at = excluded.updated_at
            """,
            (
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


def list_pending_signal_states(db_path: str, deliver_on_or_before: str) -> list[sqlite3.Row]:
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT *
                FROM signal_state
                WHERE pending_delivery = 1
                  AND deliver_on IS NOT NULL
                  AND deliver_on <= ?
                ORDER BY deliver_on, symbol
                """,
                (deliver_on_or_before,),
            ).fetchall()
        )


def get_job_state(db_path: str, job_name: str) -> sqlite3.Row | None:
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM job_state WHERE job_name = ?",
            (job_name,),
        ).fetchone()


def set_job_state(db_path: str, job_name: str, marker: str) -> None:
    updated_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO job_state (job_name, last_run_marker, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(job_name) DO UPDATE SET
                last_run_marker = excluded.last_run_marker,
                updated_at = excluded.updated_at
            """,
            (job_name, marker, updated_at),
        )


def add_trade_record(
    db_path: str,
    symbol: str,
    side: str,
    price: float,
    quantity: int,
    traded_at: str,
    note: str,
) -> None:
    """Persist a manual buy or sell record entered by the user."""

    created_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO trade_record (symbol, side, price, quantity, traded_at, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, side, price, quantity, traded_at, note, created_at),
        )


def list_trade_records(db_path: str, symbol: str | None = None) -> list[sqlite3.Row]:
    """Load trade journal entries, newest first by trade time."""

    query = """
        SELECT id, symbol, side, price, quantity, traded_at, note, created_at
        FROM trade_record
    """
    parameters: list[str] = []
    if symbol:
        query += " WHERE symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY traded_at DESC, id DESC"
    with get_connection(db_path) as connection:
        return list(connection.execute(query, parameters).fetchall())


def list_trade_records_for_symbol(db_path: str, symbol: str) -> list[sqlite3.Row]:
    """Load trade records oldest first for position reconstruction."""

    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, symbol, side, price, quantity, traded_at, note, created_at
                FROM trade_record
                WHERE symbol = ?
                ORDER BY traded_at ASC, id ASC
                """,
                (symbol,),
            ).fetchall()
        )


def add_trade_analysis(
    db_path: str,
    symbol: str,
    analysis_provider: str,
    model_name: str,
    position_summary: dict[str, Any],
    market_snapshot: dict[str, Any],
    analysis_json: dict[str, Any],
    status: str,
    error_message: str | None,
) -> None:
    """Store every LLM or rule-based trade analysis result for auditability."""

    created_at = datetime.now(UTC).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO trade_analysis (
                symbol, analysis_provider, model_name, position_summary, market_snapshot,
                analysis_json, status, error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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


def get_latest_trade_analysis(db_path: str, symbol: str | None = None) -> sqlite3.Row | None:
    """Return the latest saved trade analysis, optionally for one symbol only."""

    query = """
        SELECT id, symbol, analysis_provider, model_name, position_summary, market_snapshot,
               analysis_json, status, error_message, created_at
        FROM trade_analysis
    """
    parameters: list[str] = []
    if symbol:
        query += " WHERE symbol = ?"
        parameters.append(symbol)
    query += " ORDER BY created_at DESC, id DESC LIMIT 1"
    with get_connection(db_path) as connection:
        return connection.execute(query, parameters).fetchone()
