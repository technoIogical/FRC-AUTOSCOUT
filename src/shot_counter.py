from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import supervision as sv
from loguru import logger


@dataclass
class ShotEvent:
    frame_index: int
    timestamp: float
    team_label: str | None
    tracker_id: int | None
    ball_position: tuple[float, float]
    scored: bool | None = None  # None = unknown


@dataclass
class ScoringZone:
    """Axis-aligned rectangle in field coordinates (metres)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str = ""

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class ShotCounter:
    """Detects ball-shooting events and counts them per robot.

    Detection heuristic
    -------------------
    A *shot* is registered when a tracked ball enters a user-defined
    scoring zone **or** disappears from tracking (the ball left the
    frame / was scored).  The nearest robot at the time of the event
    is credited.
    """

    def __init__(
        self,
        scoring_zones: Sequence[ScoringZone] | None = None,
        disappear_frames: int = 5,
    ):
        self._zones = list(scoring_zones or [])
        self._disappear_frames = disappear_frames

        self._ball_last_seen: dict[int, int] = {}
        self._ball_last_pos: dict[int, np.ndarray] = {}
        self._shot_events: list[ShotEvent] = []
        self._counts: dict[str, int] = {}  # team_label → count

    # ── Public API ───────────────────────────────────────────

    @property
    def events(self) -> list[ShotEvent]:
        return list(self._shot_events)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def add_zone(self, zone: ScoringZone) -> None:
        self._zones.append(zone)
        logger.info("Added scoring zone '{}': ({},{})-({},{})",
                     zone.name, zone.x_min, zone.y_min, zone.x_max, zone.y_max)

    def clear_zones(self) -> None:
        self._zones.clear()

    def reset(self) -> None:
        self._ball_last_seen.clear()
        self._ball_last_pos.clear()
        self._shot_events.clear()
        self._counts.clear()

    # ── Per-frame update ─────────────────────────────────────

    def update(
        self,
        frame_index: int,
        timestamp: float,
        ball_detections: sv.Detections,
        ball_field_positions: np.ndarray,
        robot_detections: sv.Detections,
        robot_field_positions: np.ndarray,
        team_map: dict[int, str],
    ) -> list[ShotEvent]:
        """Process one frame and return any new shot events."""
        new_events: list[ShotEvent] = []

        current_ball_ids = set()
        if ball_detections.tracker_id is not None:
            for i, tid in enumerate(ball_detections.tracker_id):
                current_ball_ids.add(int(tid))
                pos = ball_field_positions[i]
                self._ball_last_seen[int(tid)] = frame_index
                self._ball_last_pos[int(tid)] = pos

                for zone in self._zones:
                    if zone.contains(float(pos[0]), float(pos[1])):
                        evt = self._create_event(
                            frame_index, timestamp, pos,
                            robot_detections, robot_field_positions,
                            team_map, scored=True,
                        )
                        new_events.append(evt)

        disappeared = [
            tid for tid, last_frame in self._ball_last_seen.items()
            if tid not in current_ball_ids
            and (frame_index - last_frame) == self._disappear_frames
        ]
        for tid in disappeared:
            pos = self._ball_last_pos.get(tid, np.array([0.0, 0.0]))
            evt = self._create_event(
                frame_index, timestamp, pos,
                robot_detections, robot_field_positions,
                team_map, scored=None,
            )
            new_events.append(evt)

        for evt in new_events:
            self._shot_events.append(evt)
            if evt.team_label:
                self._counts[evt.team_label] = self._counts.get(evt.team_label, 0) + 1
            logger.debug("Shot event: frame={} team={}", evt.frame_index, evt.team_label)

        return new_events

    # ── Internal ─────────────────────────────────────────────

    def _create_event(
        self,
        frame_index: int,
        timestamp: float,
        ball_pos: np.ndarray,
        robot_dets: sv.Detections,
        robot_field_pos: np.ndarray,
        team_map: dict[int, str],
        scored: bool | None,
    ) -> ShotEvent:
        nearest_tid, nearest_label = self._find_nearest_robot(
            ball_pos, robot_dets, robot_field_pos, team_map,
        )
        return ShotEvent(
            frame_index=frame_index,
            timestamp=timestamp,
            team_label=nearest_label,
            tracker_id=nearest_tid,
            ball_position=(float(ball_pos[0]), float(ball_pos[1])),
            scored=scored,
        )

    @staticmethod
    def _find_nearest_robot(
        ball_pos: np.ndarray,
        robot_dets: sv.Detections,
        robot_field_pos: np.ndarray,
        team_map: dict[int, str],
    ) -> tuple[int | None, str | None]:
        if len(robot_dets) == 0 or robot_dets.tracker_id is None:
            return None, None

        dists = np.linalg.norm(robot_field_pos - ball_pos, axis=1)
        idx = int(np.argmin(dists))
        tid = int(robot_dets.tracker_id[idx])
        return tid, team_map.get(tid)
