from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from config import (
    TRACK_THRESH,
    TRACK_BUFFER,
    MATCH_THRESH,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
)


@dataclass
class TeamAssignment:
    """Maps a ByteTrack tracker_id to a human-readable team label."""
    tracker_id: int
    team_label: str  # e.g. "Red1", "Blue3"


class FieldHomography:
    """Pixel ↔ field-coordinate mapping via perspective transform.

    Users supply four pixel-corner points that correspond to the
    four corners of the FRC field (in metres).
    """

    # Default field corners in metres (top-left origin, x→right, y→down)
    DEFAULT_FIELD_CORNERS = np.array([
        [0.0, 0.0],
        [FIELD_LENGTH_M, 0.0],
        [FIELD_LENGTH_M, FIELD_WIDTH_M],
        [0.0, FIELD_WIDTH_M],
    ], dtype=np.float32)

    def __init__(self) -> None:
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    def calibrate(
        self,
        pixel_corners: np.ndarray,
        field_corners: np.ndarray | None = None,
    ) -> None:
        """Compute homography from four pixel ↔ field corner pairs.

        Parameters
        ----------
        pixel_corners : (4, 2) float array of pixel coordinates.
        field_corners : (4, 2) float array of field coordinates in metres.
                        Defaults to a standard FRC field rectangle.
        """
        if field_corners is None:
            field_corners = self.DEFAULT_FIELD_CORNERS

        pixel_corners = np.asarray(pixel_corners, dtype=np.float32)
        field_corners = np.asarray(field_corners, dtype=np.float32)

        self._H, status = cv2.findHomography(pixel_corners, field_corners)
        if self._H is None:
            raise ValueError("Homography computation failed – check corner points")
        self._H_inv = np.linalg.inv(self._H)
        logger.info("Field homography calibrated")

    def pixel_to_field(self, points: np.ndarray) -> np.ndarray:
        """Convert (N, 2) pixel coords → (N, 2) field coords in metres."""
        if not self.is_calibrated:
            raise RuntimeError("Homography not calibrated – call calibrate() first")
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._H)
        return transformed.reshape(-1, 2)

    def field_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Convert (N, 2) field coords → (N, 2) pixel coords."""
        if not self.is_calibrated:
            raise RuntimeError("Homography not calibrated")
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self._H_inv)
        return transformed.reshape(-1, 2)


class RobotTracker:
    """Multi-object tracker with team-ID assignment and coordinate mapping."""

    def __init__(self, homography: FieldHomography | None = None):
        self._byte_track = sv.ByteTrack(
            track_activation_threshold=TRACK_THRESH,
            lost_track_buffer=TRACK_BUFFER,
            minimum_matching_threshold=MATCH_THRESH,
            frame_rate=30,
        )
        self.homography = homography or FieldHomography()
        self._team_map: dict[int, str] = {}

    # ── Team assignment ──────────────────────────────────────

    def assign_team(self, tracker_id: int, team_label: str) -> None:
        self._team_map[tracker_id] = team_label
        logger.info("Assigned tracker {} → {}", tracker_id, team_label)

    def get_team(self, tracker_id: int) -> str | None:
        return self._team_map.get(tracker_id)

    @property
    def team_map(self) -> dict[int, str]:
        return dict(self._team_map)

    # ── Tracking ─────────────────────────────────────────────

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Feed detections through ByteTrack and return tracked detections.

        The returned object has ``tracker_id`` populated.
        """
        tracked = self._byte_track.update_with_detections(detections)
        return tracked

    def reset(self) -> None:
        self._byte_track.reset()
        self._team_map.clear()
        logger.info("Tracker reset")

    # ── Coordinate helpers ───────────────────────────────────

    def detection_centres(self, detections: sv.Detections) -> np.ndarray:
        """Return (N, 2) bottom-centre points of bounding boxes."""
        boxes = detections.xyxy  # (N, 4)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = boxes[:, 3]  # bottom edge ≈ foot position
        return np.stack([cx, cy], axis=1)

    def pixel_to_field(self, pixel_points: np.ndarray) -> np.ndarray:
        return self.homography.pixel_to_field(pixel_points)
