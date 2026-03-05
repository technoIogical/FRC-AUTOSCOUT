from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
from loguru import logger

from config import DUCKDB_PATH

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id   VARCHAR PRIMARY KEY,
    event      VARCHAR,
    match_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS robot_tracks (
    id         INTEGER DEFAULT (nextval('robot_tracks_seq')),
    match_id   VARCHAR NOT NULL,
    team_label VARCHAR NOT NULL,
    frame      INTEGER NOT NULL,
    x          DOUBLE  NOT NULL,
    y          DOUBLE  NOT NULL,
    ts         DOUBLE  NOT NULL
);

CREATE TABLE IF NOT EXISTS ball_tracks (
    id       INTEGER DEFAULT (nextval('ball_tracks_seq')),
    match_id VARCHAR NOT NULL,
    frame    INTEGER NOT NULL,
    x        DOUBLE  NOT NULL,
    y        DOUBLE  NOT NULL,
    z        DOUBLE  DEFAULT 0.0,
    ts       DOUBLE  NOT NULL
);

CREATE TABLE IF NOT EXISTS shot_events (
    id         INTEGER DEFAULT (nextval('shot_events_seq')),
    match_id   VARCHAR NOT NULL,
    team_label VARCHAR,
    frame      INTEGER NOT NULL,
    ts         DOUBLE  NOT NULL,
    ball_x     DOUBLE,
    ball_y     DOUBLE,
    scored     BOOLEAN
);
"""


class LocalDB:
    """Thin wrapper around DuckDB for match / tracking data."""

    def __init__(self, db_path: str | Path = DUCKDB_PATH):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        self._init_schema()
        logger.info("DuckDB opened: {}", db_path)

    # ── Schema ───────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._con.execute("CREATE SEQUENCE IF NOT EXISTS robot_tracks_seq START 1")
        self._con.execute("CREATE SEQUENCE IF NOT EXISTS ball_tracks_seq START 1")
        self._con.execute("CREATE SEQUENCE IF NOT EXISTS shot_events_seq START 1")
        self._con.execute(_SCHEMA_SQL)

    # ── Matches ──────────────────────────────────────────────

    def create_match(self, match_id: str, event: str = "") -> None:
        self._con.execute(
            "INSERT OR IGNORE INTO matches (match_id, event) VALUES (?, ?)",
            [match_id, event],
        )

    # ── Bulk inserts ─────────────────────────────────────────

    def insert_robot_tracks(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._con.executemany(
            "INSERT INTO robot_tracks (match_id, team_label, frame, x, y, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(r["match_id"], r["team_label"], r["frame"], r["x"], r["y"], r["ts"]) for r in rows],
        )

    def insert_ball_tracks(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._con.executemany(
            "INSERT INTO ball_tracks (match_id, frame, x, y, z, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(r["match_id"], r["frame"], r["x"], r["y"], r.get("z", 0.0), r["ts"]) for r in rows],
        )

    def insert_shot_events(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._con.executemany(
            "INSERT INTO shot_events (match_id, team_label, frame, ts, ball_x, ball_y, scored) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (r["match_id"], r.get("team_label"), r["frame"], r["ts"],
                 r.get("ball_x"), r.get("ball_y"), r.get("scored"))
                for r in rows
            ],
        )

    # ── Queries ──────────────────────────────────────────────

    def query(self, sql: str, params: list | None = None) -> list[tuple]:
        return self._con.execute(sql, params or []).fetchall()

    def query_df(self, sql: str, params: list | None = None):
        """Return query result as a Pandas DataFrame."""
        return self._con.execute(sql, params or []).fetchdf()

    def list_matches(self) -> list[tuple]:
        return self.query("SELECT match_id, event, match_time FROM matches ORDER BY match_time DESC")

    # ── Lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()
        logger.info("DuckDB connection closed")

    def __enter__(self) -> LocalDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
