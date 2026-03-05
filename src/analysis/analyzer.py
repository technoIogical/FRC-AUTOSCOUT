from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger

from src.db.local_db import LocalDB


class MatchAnalyzer:
    """High-level analytics on match tracking data."""

    def __init__(self, db: LocalDB):
        self._db = db

    # ── Polars-based fast queries ────────────────────────────

    def robot_track_polars(self, match_id: str) -> pl.DataFrame:
        pdf = self._db.query_df(
            "SELECT team_label, frame, x, y, ts FROM robot_tracks WHERE match_id = ?",
            [match_id],
        )
        return pl.from_pandas(pdf)

    def ball_track_polars(self, match_id: str) -> pl.DataFrame:
        pdf = self._db.query_df(
            "SELECT frame, x, y, z, ts FROM ball_tracks WHERE match_id = ?",
            [match_id],
        )
        return pl.from_pandas(pdf)

    def shot_events_polars(self, match_id: str) -> pl.DataFrame:
        pdf = self._db.query_df(
            "SELECT team_label, frame, ts, ball_x, ball_y, scored FROM shot_events WHERE match_id = ?",
            [match_id],
        )
        return pl.from_pandas(pdf)

    # ── Summary statistics ───────────────────────────────────

    def shot_summary(self, match_id: str) -> dict[str, Any]:
        df = self.shot_events_polars(match_id)
        if df.is_empty():
            return {}
        summary = (
            df.group_by("team_label")
            .agg([
                pl.count().alias("total_shots"),
                pl.col("scored").sum().alias("scored_count"),
            ])
            .to_dicts()
        )
        return {row["team_label"]: row for row in summary}

    def heatmap_data(self, match_id: str, team_label: str | None = None) -> pd.DataFrame:
        """Return robot positions as a Pandas DataFrame for heatmap rendering."""
        sql = "SELECT team_label, x, y FROM robot_tracks WHERE match_id = ?"
        params: list = [match_id]
        if team_label:
            sql += " AND team_label = ?"
            params.append(team_label)
        return self._db.query_df(sql, params)

    def field_coverage(self, match_id: str) -> dict[str, float]:
        """Approximate field coverage per team (fraction of 1m x 1m cells visited)."""
        df = self.robot_track_polars(match_id)
        if df.is_empty():
            return {}
        result = {}
        for label in df["team_label"].unique().to_list():
            sub = df.filter(pl.col("team_label") == label)
            cells = set(
                zip(sub["x"].cast(pl.Int32).to_list(), sub["y"].cast(pl.Int32).to_list())
            )
            from config import FIELD_LENGTH_M, FIELD_WIDTH_M
            total_cells = int(FIELD_LENGTH_M) * int(FIELD_WIDTH_M)
            result[label] = len(cells) / total_cells if total_cells else 0.0
        return result

    # ── ydata-profiling report ───────────────────────────────

    def generate_profile_report(self, match_id: str, output_path: Path | str | None = None) -> Path:
        from ydata_profiling import ProfileReport

        df = self._db.query_df(
            "SELECT * FROM robot_tracks WHERE match_id = ?", [match_id],
        )
        if output_path is None:
            output_path = Path(f"data/report_{match_id}.html")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = ProfileReport(df, title=f"Match {match_id} – Robot Tracks", minimal=True)
        report.to_file(output_path)
        logger.info("Profile report saved to {}", output_path)
        return output_path
