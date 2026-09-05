"""
kiosk_agent/db.py
-----------------
SQLite (WAL mode) persistence layer for the Kiosk Agent.

Tables implemented (all DDLs sourced from architecture.md §6.2–§6.4):
  - tickets          — local ticket queue (§6.2)
  - asset_registry   — cached read-only replica (§6.3); writable only by Sync Daemon
  - sync_log         — durable sync audit trail (§6.4)
  - ticket_seq       — per-kiosk, per-day monotonic counter (implementation.md §4.5)

Encryption:
  When DB_ENCRYPTION_KEY is set (non-empty), SQLCipher is used for data-at-rest
  encryption (architecture.md §8). Otherwise plain SQLite is used (dev/CI).

WAL mode:
  Enabled on every connection to survive kiosk power loss mid-write (architecture.md
  §ADR-003, §10 failure mode: "Kiosk loses power mid-complaint").
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator

import sqlalchemy as sa
from sqlalchemy import event, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from kiosk_agent.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine() -> sa.Engine:
    """
    Create the SQLAlchemy engine.

    - If DB_ENCRYPTION_KEY is set, use pysqlcipher3 dialect for SQLCipher.
    - Otherwise use the standard sqlite+pysqlite dialect (plain SQLite).
    - WAL mode and foreign-key enforcement are applied via a connect event.
    """
    db_path = str(settings.db_path.resolve())
    encryption_key = settings.db_encryption_key

    if encryption_key:
        # SQLCipher — requires sqlcipher3-binary installed (pyproject.toml).
        # The connection string for pysqlcipher3 uses the 'sqlite+pysqlcipher' dialect.
        # The key pragma is injected via creator= so it never appears in the URL.
        try:
            import pysqlcipher3.dbapi2 as sqlcipher  # noqa: PLC0415

            def _creator() -> sqlcipher.Connection:  # type: ignore[name-defined]
                conn = sqlcipher.connect(db_path)
                conn.execute(f"PRAGMA key='{encryption_key}';")
                return conn

            engine = sa.create_engine(
                "sqlite+pysqlcipher://",
                creator=_creator,
                echo=settings.log_level == "debug",
            )
            logger.info("Database: SQLCipher encrypted at %s", db_path)
        except ImportError:
            logger.warning(
                "pysqlcipher3 not available; falling back to plain SQLite. "
                "Set DB_ENCRYPTION_KEY only in environments where sqlcipher3-binary "
                "is installed."
            )
            engine = _plain_sqlite_engine(db_path)
    else:
        engine = _plain_sqlite_engine(db_path)
        logger.info("Database: plain SQLite (unencrypted) at %s", db_path)

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: object, _record: object) -> None:
        """Apply per-connection pragmas immediately after each new connection."""
        cursor = dbapi_conn.cursor()  # type: ignore[union-attr]
        cursor.execute("PRAGMA journal_mode=WAL;")   # crash-safe durability
        cursor.execute("PRAGMA foreign_keys=ON;")    # enforce FK constraints
        cursor.execute("PRAGMA synchronous=NORMAL;") # balanced durability/speed
        cursor.close()

    return engine


def _plain_sqlite_engine(db_path: str) -> sa.Engine:
    return sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=settings.log_level == "debug",
    )


# ---------------------------------------------------------------------------
# ORM base and models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class TicketRow(Base):
    """
    Local SQLite ticket queue (architecture.md §6.2).
    `structured_complaint` is stored as JSON text; the Pydantic model
    is the canonical shape — this column is never queried by field.
    """
    __tablename__ = "tickets"

    ticket_id:               Mapped[str]         = mapped_column(sa.Text, primary_key=True)
    kiosk_id:                Mapped[str]         = mapped_column(sa.Text, nullable=False)
    created_at:              Mapped[str]         = mapped_column(sa.Text, nullable=False)  # ISO-8601 UTC
    language:                Mapped[str]         = mapped_column(sa.Text, nullable=False)
    raw_transcript:          Mapped[str]         = mapped_column(sa.Text, nullable=False)
    structured_complaint:    Mapped[str]         = mapped_column(sa.Text, nullable=False)  # JSON
    category:                Mapped[str]         = mapped_column(sa.Text, nullable=False)
    department_code:         Mapped[str]         = mapped_column(sa.Text, nullable=False)
    asset_id:                Mapped[str | None]  = mapped_column(sa.Text, nullable=True)
    location_lat:            Mapped[float | None]= mapped_column(sa.Float, nullable=True)
    location_lon:            Mapped[float | None]= mapped_column(sa.Float, nullable=True)
    evidence_image_path:     Mapped[str | None]  = mapped_column(sa.Text, nullable=True)
    evidence_image_hash:     Mapped[str | None]  = mapped_column(sa.Text, nullable=True)
    verification_confidence: Mapped[float | None]= mapped_column(sa.Float, nullable=True)
    evidence_status:         Mapped[str]         = mapped_column(sa.Text, nullable=False)
    urgency_score:           Mapped[float]       = mapped_column(sa.Float, nullable=False)
    status:                  Mapped[str]         = mapped_column(sa.Text, nullable=False, default="QUEUED")
    sync_attempts:           Mapped[int]         = mapped_column(sa.Integer, nullable=False, default=0)
    last_sync_attempt:       Mapped[str | None]  = mapped_column(sa.Text, nullable=True)

    # Index: Sync Daemon needs QUEUED tickets ordered by urgency (architecture.md §4.7)
    __table_args__ = (
        sa.Index("idx_tickets_status_urgency", "status", "urgency_score"),
        sa.Index("idx_tickets_asset_created", "asset_id", "created_at"),
    )


class AssetRegistryRow(Base):
    """
    Cached read-only asset registry replica (architecture.md §6.3).
    Writable only by the Sync Daemon during a registry-delta pull.
    geo_polygon is stored as GeoJSON text; spatial filtering is done
    in Python via Shapely (no PostGIS on edge).
    """
    __tablename__ = "asset_registry"

    asset_id:        Mapped[str] = mapped_column(sa.Text, primary_key=True)
    asset_type:      Mapped[str] = mapped_column(sa.Text, nullable=False)
    department_code: Mapped[str] = mapped_column(sa.Text, nullable=False)
    district_code:   Mapped[str] = mapped_column(sa.Text, nullable=False)
    # GeoJSON Polygon, WGS84 — matched in Python via Shapely.contains()
    geo_polygon:     Mapped[str] = mapped_column(sa.Text, nullable=False)
    # Human-readable label for fuzzy landmark resolution (architecture.md §4.3)
    asset_label:     Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    # Centroid pre-computed at seed/sync time to speed fuzzy matching
    centroid_lat:    Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    centroid_lon:    Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    registry_version:Mapped[int] = mapped_column(sa.Integer, nullable=False)
    last_synced_at:  Mapped[str | None] = mapped_column(sa.Text, nullable=True)

    __table_args__ = (
        sa.Index("idx_asset_registry_dept", "department_code"),
        sa.Index("idx_asset_registry_version", "registry_version"),
    )


class ImageryRow(Base):
    """
    Local imagery store — before/after tile pairs keyed by asset_id.
    (architecture.md §4.4, implementation.md §4.4)
    """
    __tablename__ = "imagery"

    id:              Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    asset_id:        Mapped[str] = mapped_column(sa.Text, nullable=False, index=True)
    before_path:     Mapped[str] = mapped_column(sa.Text, nullable=False)
    after_path:      Mapped[str] = mapped_column(sa.Text, nullable=False)
    tile_version:    Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    captured_at:     Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class SyncLogRow(Base):
    """
    Durable audit trail for every sync attempt (architecture.md §6.4).
    Never deleted — provides the `sync_log` view queried by monitoring.
    """
    __tablename__ = "sync_log"

    sync_id:      Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticket_id:    Mapped[str] = mapped_column(sa.Text, nullable=False, index=True)
    sync_method:  Mapped[str] = mapped_column(sa.Text, nullable=False, default="HTTPS")
    initiated_at: Mapped[str] = mapped_column(sa.Text, nullable=False)
    completed_at: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    result:       Mapped[str] = mapped_column(sa.Text, nullable=False)  # SUCCESS|FAILED|CONFLICT
    error_detail: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class TicketSeqRow(Base):
    """
    Per-kiosk, per-day monotonic counter for ticket_id generation.
    (implementation.md §4.5 — "incremented atomically within the same
    SQLite transaction as the ticket insert")
    """
    __tablename__ = "ticket_seq"

    kiosk_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    day:      Mapped[str] = mapped_column(sa.Text, primary_key=True)  # YYYYMMDD
    seq:      Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)


# ---------------------------------------------------------------------------
# Engine + session factory (module-level singletons)
# ---------------------------------------------------------------------------

_engine: sa.Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> sa.Engine:
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a SQLAlchemy Session and commits on clean exit
    or rolls back on exception.

    Usage:
        with db_session() as session:
            session.add(row)
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables if they do not already exist.
    Called once at Kiosk Agent startup (main.py lifespan hook).
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS semantics via
    SQLAlchemy's checkfirst=True.
    """
    engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)
    logger.info("Database schema initialised (WAL mode active).")


# ---------------------------------------------------------------------------
# Helper utilities used by Ticket Router and Sync Daemon
# ---------------------------------------------------------------------------

def next_ticket_seq(session: Session, kiosk_id: str) -> int:
    """
    Atomically increment and return the next per-kiosk, per-day sequence number.
    Must be called within an open transaction (i.e. inside db_session()).

    This is the implementation of the atomicity guarantee described in
    implementation.md §4.5: "incremented atomically within the same SQLite
    transaction as the ticket insert, preventing ID collisions under concurrent
    finalize calls."
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    row = session.get(TicketSeqRow, {"kiosk_id": kiosk_id, "day": today})
    if row is None:
        row = TicketSeqRow(kiosk_id=kiosk_id, day=today, seq=1)
        session.add(row)
    else:
        row.seq += 1
    return row.seq


def get_storage_pct_used(db_path: Path | None = None) -> float:
    """
    Return the percentage of the filesystem's total capacity used by the
    database file's partition. Used by GET /v1/health and the UI status indicator.
    Falls back to 0.0 if the path cannot be stat'd (e.g. in-memory test DBs).
    """
    path = db_path or settings.db_path
    try:
        stat = os.statvfs(path.resolve().parent)
        total = stat.f_blocks * stat.f_frsize
        free  = stat.f_bfree  * stat.f_frsize
        used  = total - free
        return round((used / total) * 100, 1) if total > 0 else 0.0
    except (AttributeError, ZeroDivisionError, FileNotFoundError):
        # os.statvfs is POSIX-only; on Windows dev machines return a placeholder.
        try:
            import shutil  # noqa: PLC0415
            total, _used_bytes, free = shutil.disk_usage(path.resolve().parent)
            used = total - free
            return round((used / total) * 100, 1) if total > 0 else 0.0
        except Exception:
            return 0.0


def queued_ticket_count(session: Session) -> int:
    """Return the number of tickets with status=QUEUED (unsynced)."""
    return session.execute(
        text("SELECT COUNT(*) FROM tickets WHERE status = 'QUEUED'")
    ).scalar_one()
