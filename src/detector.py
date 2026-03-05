from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import supervision as sv
from ultralytics import YOLO
from loguru import logger

from config import (
    ROBOT_MODEL_PATH,
    BALL_MODEL_PATH,
    ROBOT_CONFIDENCE,
    BALL_CONFIDENCE,
)


class RobotBallDetector:
    """Wraps two YOLO models (robot + ball) and merges detections."""

    CLASS_ROBOT = 0
    CLASS_BALL = 1

    def __init__(
        self,
        robot_model_path: Union[str, Path] = ROBOT_MODEL_PATH,
        ball_model_path: Union[str, Path] = BALL_MODEL_PATH,
        robot_conf: float = ROBOT_CONFIDENCE,
        ball_conf: float = BALL_CONFIDENCE,
    ):
        logger.info("Loading robot model: {}", robot_model_path)
        self._robot_model = YOLO(str(robot_model_path))

        logger.info("Loading ball model: {}", ball_model_path)
        self._ball_model = YOLO(str(ball_model_path))

        self._robot_conf = robot_conf
        self._ball_conf = ball_conf
        logger.info(
            "Detector ready (robot_conf={}, ball_conf={})",
            robot_conf, ball_conf,
        )

    # ── Core detection ───────────────────────────────────────

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run both models on *frame* and return merged sv.Detections.

        The returned detections carry a ``class_id`` array where
        ``0`` = robot, ``1`` = ball.
        """
        robot_dets = self._run_model(
            self._robot_model, frame, self._robot_conf, self.CLASS_ROBOT,
        )
        ball_dets = self._run_model(
            self._ball_model, frame, self._ball_conf, self.CLASS_BALL,
        )
        merged = sv.Detections.merge([robot_dets, ball_dets])
        if merged.class_id is None:
            merged.class_id = np.array([], dtype=int)
        return merged

    def detect_robots(self, frame: np.ndarray) -> sv.Detections:
        return self._run_model(
            self._robot_model, frame, self._robot_conf, self.CLASS_ROBOT,
        )

    def detect_balls(self, frame: np.ndarray) -> sv.Detections:
        return self._run_model(
            self._ball_model, frame, self._ball_conf, self.CLASS_BALL,
        )

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _run_model(
        model: YOLO,
        frame: np.ndarray,
        conf: float,
        class_id: int,
    ) -> sv.Detections:
        results = model(frame, conf=conf, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        if len(dets) > 0:
            dets.class_id = np.full(len(dets), class_id, dtype=int)
        return dets
