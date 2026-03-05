from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Union

import cv2
import numpy as np
from loguru import logger


@dataclass(frozen=True)
class FrameMeta:
    fps: float
    width: int
    height: int
    total_frames: int | None  # None for live camera


class VideoSource:
    """Unified video source supporting camera and file inputs."""

    def __init__(self, cap: cv2.VideoCapture, name: str, is_live: bool = False):
        self._cap = cap
        self.name = name
        self.is_live = is_live

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {name}")

        self.meta = FrameMeta(
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=(
                None if is_live
                else int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ),
        )
        logger.info(
            "VideoSource opened: {} ({}x{} @ {:.1f}fps)",
            name, self.meta.width, self.meta.height, self.meta.fps,
        )

    # ── Factory methods ──────────────────────────────────────

    @classmethod
    def from_camera(cls, device_id: int = 0) -> VideoSource:
        cap = cv2.VideoCapture(device_id)
        return cls(cap, name=f"camera:{device_id}", is_live=True)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> VideoSource:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        return cls(cap, name=path.name, is_live=False)

    # ── Frame iteration ──────────────────────────────────────

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_bgr) tuples."""
        idx = 0
        while self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        logger.info("VideoSource exhausted after {} frames", idx)

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame. Returns (success, frame_or_none)."""
        ok, frame = self._cap.read()
        return ok, frame if ok else None

    def seek(self, frame_index: int) -> None:
        if self.is_live:
            logger.warning("Cannot seek on a live camera source")
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # ── Cleanup ──────────────────────────────────────────────

    def release(self) -> None:
        self._cap.release()
        logger.info("VideoSource released: {}", self.name)

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
