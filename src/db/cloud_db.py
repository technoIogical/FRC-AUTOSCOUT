from __future__ import annotations

import os
from typing import Any, Sequence

from loguru import logger

from config import SUPABASE_URL, SUPABASE_KEY


class CloudDB:
    """Supabase sync layer – pushes local data to the cloud after a match."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
    ):
        self._url = url or SUPABASE_URL or os.getenv("SUPABASE_URL", "")
        self._key = key or SUPABASE_KEY or os.getenv("SUPABASE_KEY", "")
        self._client = None

    # ── Lazy connection ──────────────────────────────────────

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._url or not self._key:
            raise RuntimeError(
                "Supabase credentials missing. Set SUPABASE_URL / SUPABASE_KEY "
                "in config.py or as environment variables."
            )
        from supabase import create_client
        self._client = create_client(self._url, self._key)
        logger.info("Supabase client connected to {}", self._url)

    @property
    def is_configured(self) -> bool:
        url = self._url or os.getenv("SUPABASE_URL", "")
        key = self._key or os.getenv("SUPABASE_KEY", "")
        return bool(url and key)

    # ── Upsert helpers ───────────────────────────────────────

    def sync_match(self, match: dict[str, Any]) -> None:
        self._ensure_client()
        self._client.table("matches").upsert(match).execute()
        logger.info("Synced match {}", match.get("match_id"))

    def sync_robot_tracks(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._ensure_client()
        self._client.table("robot_tracks").insert(list(rows)).execute()
        logger.info("Synced {} robot track rows", len(rows))

    def sync_ball_tracks(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._ensure_client()
        self._client.table("ball_tracks").insert(list(rows)).execute()
        logger.info("Synced {} ball track rows", len(rows))

    def sync_shot_events(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._ensure_client()
        self._client.table("shot_events").insert(list(rows)).execute()
        logger.info("Synced {} shot event rows", len(rows))

    # ── Bulk sync from LocalDB ───────────────────────────────

    def sync_all_from_local(self, local_db) -> None:
        """Pull everything for the latest match from *local_db* and push it."""
        matches = local_db.list_matches()
        if not matches:
            logger.warning("No matches to sync")
            return

        match_id = matches[0][0]
        self.sync_match({"match_id": match_id, "event": matches[0][1]})

        robot_rows = local_db.query_df(
            "SELECT match_id, team_label, frame, x, y, ts FROM robot_tracks WHERE match_id = ?",
            [match_id],
        ).to_dict(orient="records")
        self.sync_robot_tracks(robot_rows)

        ball_rows = local_db.query_df(
            "SELECT match_id, frame, x, y, z, ts FROM ball_tracks WHERE match_id = ?",
            [match_id],
        ).to_dict(orient="records")
        self.sync_ball_tracks(ball_rows)

        shot_rows = local_db.query_df(
            "SELECT match_id, team_label, frame, ts, ball_x, ball_y, scored "
            "FROM shot_events WHERE match_id = ?",
            [match_id],
        ).to_dict(orient="records")
        self.sync_shot_events(shot_rows)

        logger.info("Full sync complete for match {}", match_id)
