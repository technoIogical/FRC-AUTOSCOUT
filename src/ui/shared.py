"""Shared state, constants, and utilities used by both Settings and Analysis views."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import flet as ft
import flet_video as ftv
import numpy as np
from loguru import logger

from config import (
    BASE_DIR,
    FIELD_IMAGE_PATH,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    ROBOT_SIZE_M,
    ROBOT_RED_ICON,
    ROBOT_BLUE_ICON,
)

from src.detector import RobotBallDetector
from src.tracker import RobotTracker, FieldHomography
from src.shot_counter import ShotCounter, ScoringZone
from src.db.local_db import LocalDB
from src.db.cloud_db import CloudDB
from src.analysis.analyzer import MatchAnalyzer
from src.video_source import VideoSource

# ── Display constants ────────────────────────────────────────
MAP_W = 500
MAP_H = 250
_PX_PER_M_X = MAP_W / FIELD_LENGTH_M
_PX_PER_M_Y = MAP_H / FIELD_WIDTH_M
ROBOT_ICON_PX = round(ROBOT_SIZE_M * min(_PX_PER_M_X, _PX_PER_M_Y))
BALL_DOT_PX = max(4, round(0.24 * min(_PX_PER_M_X, _PX_PER_M_Y)))

VIDEO_W, VIDEO_H = 660, 400
DETECT_EVERY_N = 6

_CALIB_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200)]
_CALIB_LABELS = ["左上", "右上", "右下", "左下"]
_DRAG_RADIUS_PX = 30


# ── AppState ─────────────────────────────────────────────────

class AppState:
    """Mutable shared state between Settings and Analysis views."""

    def __init__(self):
        self.running = False
        self.source: VideoSource | None = None
        self.detector: RobotBallDetector | None = None
        self.robot_tracker = RobotTracker()
        self.shot_counter = ShotCounter()
        self.db = LocalDB()
        self.cloud_db = CloudDB()
        self.analyzer = MatchAnalyzer(self.db)
        self.match_id = ""
        self.video_path: str = ""
        self.frame_index = 0
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        self.start_frame_img: np.ndarray | None = None
        self.robot_field_positions: dict[int, tuple[float, float]] = {}
        self.ball_field_positions: list[tuple[float, float]] = []

        self.calibration_mode = False
        self.calibration_pixels: list[tuple[float, float]] = []

        self.zone_draw_mode = False
        self.zone_start: tuple[float, float] | None = None

        self.frame_positions: dict[int, dict] = {}

    @property
    def is_ready(self) -> bool:
        """All required settings are configured for processing."""
        return (
            self.source is not None
            and bool(self.video_path)
            and self.start_frame_img is not None
        )


# ── Utility functions ────────────────────────────────────────

def frame_to_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"


def draw_calib_overlay(frame: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """Draw calibration markers and connecting lines on a copy of *frame*."""
    vis = frame.copy()
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        if n >= 2 and (i + 1 < n or n == 4):
            cv2.line(vis,
                     (int(points[i][0]), int(points[i][1])),
                     (int(points[j][0]), int(points[j][1])),
                     (0, 255, 255), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(points):
        c = (int(px), int(py))
        cv2.circle(vis, c, 12, _CALIB_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(vis, c, 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, _CALIB_LABELS[i], (c[0] + 18, c[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def make_player(playlist=None):
    return ftv.Video(
        width=VIDEO_W, height=VIDEO_H,
        fill_color=ft.Colors.BLACK,
        aspect_ratio=16 / 9,
        autoplay=bool(playlist),
        show_controls=True,
        filter_quality=ft.FilterQuality.HIGH,
        playlist=playlist or [],
        on_load=lambda e: logger.info("Video player loaded"),
        on_error=lambda e: logger.error("Video error: {}", getattr(e, "data", e)),
    )


def update_field_overlay(state: AppState, field_stack: ft.Stack):
    """Rebuild field map overlay: scoring zones + robots + balls."""
    field_stack.controls = [field_stack.controls[0]]

    for zone in state.shot_counter._zones:
        zl = zone.x_min / FIELD_LENGTH_M * MAP_W
        zt = zone.y_min / FIELD_WIDTH_M * MAP_H
        zw = (zone.x_max - zone.x_min) / FIELD_LENGTH_M * MAP_W
        zh = (zone.y_max - zone.y_min) / FIELD_WIDTH_M * MAP_H
        field_stack.controls.append(
            ft.Container(
                content=ft.Text(zone.name, size=10, color=ft.Colors.WHITE,
                                text_align=ft.TextAlign.CENTER),
                width=max(zw, 1), height=max(zh, 1),
                bgcolor=ft.Colors.GREEN, opacity=0.35,
                border=ft.border.all(2, ft.Colors.LIGHT_GREEN),
                left=zl, top=zt,
                alignment=ft.Alignment(0, 0),
            )
        )

    half = ROBOT_ICON_PX / 2
    for tid, (fx, fy) in state.robot_field_positions.items():
        px = fx / FIELD_LENGTH_M * MAP_W
        py = fy / FIELD_WIDTH_M * MAP_H
        team = state.robot_tracker.get_team(tid)
        is_red = team and "Red" in team
        icon_path = str(ROBOT_RED_ICON if is_red else ROBOT_BLUE_ICON)
        field_stack.controls.append(
            ft.Container(
                content=ft.Image(src=icon_path, width=ROBOT_ICON_PX, height=ROBOT_ICON_PX, fit="contain"),
                left=max(0, px - half), top=max(0, py - half),
                tooltip=f"#{tid} {team or ''}",
            )
        )

    ball_half = BALL_DOT_PX / 2
    for fx, fy in state.ball_field_positions:
        px = fx / FIELD_LENGTH_M * MAP_W
        py = fy / FIELD_WIDTH_M * MAP_H
        field_stack.controls.append(
            ft.Container(
                width=BALL_DOT_PX, height=BALL_DOT_PX,
                bgcolor=ft.Colors.YELLOW, border_radius=BALL_DOT_PX / 2,
                left=max(0, px - ball_half), top=max(0, py - ball_half),
                tooltip="Ball",
            )
        )


# ── Batch processing (runs in background thread) ────────────

def batch_process(
    state: AppState,
    page: ft.Page,
    cv_source: VideoSource,
    output_path: str,
    live_image: ft.Image,
    field_stack: ft.Stack,
    fps_text: ft.Text,
    frame_text: ft.Text,
    shot_table: ft.DataTable,
    progress_bar: ft.ProgressBar,
    progress_text: ft.Text,
    process_btn: ft.Button,
    status_text: ft.Text,
    video_container: ft.Container,
    add_log,
):
    import supervision as sv

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    src_fps = cv_source.meta.fps or 30.0
    w = cv_source.meta.width
    h = cv_source.meta.height
    start = state.start_frame
    end = state.end_frame if state.end_frame > 0 else (cv_source.meta.total_frames or 0)
    total_to_process = max(end - start, 1)

    out_p = Path(output_path)
    if out_p.exists():
        out_p.unlink()
        logger.info("Deleted existing output: {}", output_path)

    writer = None
    for codec in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        ext = ".avi" if codec == "XVID" else ".mp4"
        out = str(Path(output_path).with_suffix(ext))
        if Path(out).exists():
            Path(out).unlink()
        w_try = cv2.VideoWriter(out, fourcc, src_fps, (w, h))
        if w_try.isOpened():
            writer = w_try
            output_path = out
            logger.info("VideoWriter opened with codec={}, path={}", codec, out)
            break
        w_try.release()
    if writer is None:
        logger.error("No working VideoWriter codec found for: {}", output_path)
        add_log("错误：无法创建视频写入器，请检查 OpenCV 编解码器")
        state.running = False
        return

    robot_track_buffer: list[dict] = []
    ball_track_buffer: list[dict] = []
    FLUSH_INTERVAL = 30
    detect_count = 0
    last_ui_update = 0.0

    logger.info("Batch processing: frames {}-{} → {}", start, end, output_path)

    cv_source.seek(start)
    verify_ok, verify_frame = cv_source.read_frame()
    if not verify_ok:
        logger.error("Cannot read frame {} after seek — video may not support seeking", start)
        add_log(f"错误：无法读取第 {start} 帧，尝试顺序读取…")
        cv_source.release()
        cv_source = VideoSource.from_file(state.video_path)
        for _ in range(start):
            cv_source.read_frame()
        verify_ok, verify_frame = cv_source.read_frame()
        if not verify_ok:
            logger.error("Sequential read also failed at frame {}", start)
            add_log("错误：视频读取失败")
            state.running = False
            writer.release()
            return
    cv_source.seek(start)

    frames_read = 0
    for frame_idx in range(start, end):
        if not state.running:
            break

        ok, frame = cv_source.read_frame()
        if not ok or frame is None:
            logger.warning("Read failed at frame {}, stopping", frame_idx)
            break
        frames_read += 1

        state.frame_index = frame_idx
        timestamp = frame_idx / src_fps
        annotated = frame
        run_detect = (frame_idx % DETECT_EVERY_N == 0)

        if run_detect:
            t0 = time.perf_counter()
            detect_count += 1
            try:
                all_dets = state.detector.detect(frame)

                if len(all_dets) > 0 and all_dets.class_id is not None:
                    robot_mask = all_dets.class_id == RobotBallDetector.CLASS_ROBOT
                    ball_mask = all_dets.class_id == RobotBallDetector.CLASS_BALL
                    robot_dets = all_dets[robot_mask]
                    ball_dets = all_dets[ball_mask]
                else:
                    robot_dets = sv.Detections.empty()
                    ball_dets = sv.Detections.empty()

                tracked_robots = state.robot_tracker.update(robot_dets)
                robot_centres = state.robot_tracker.detection_centres(tracked_robots)
                ball_centres = (
                    state.robot_tracker.detection_centres(ball_dets) if len(ball_dets) > 0
                    else np.empty((0, 2))
                )

                robot_field = np.empty((0, 2))
                ball_field = np.empty((0, 2))

                if state.robot_tracker.homography.is_calibrated:
                    if len(robot_centres) > 0:
                        robot_field = state.robot_tracker.pixel_to_field(robot_centres)
                    if len(ball_centres) > 0:
                        ball_field = state.robot_tracker.pixel_to_field(ball_centres)

                    rpos = {}
                    if tracked_robots.tracker_id is not None:
                        for i, tid in enumerate(tracked_robots.tracker_id):
                            rpos[int(tid)] = (float(robot_field[i][0]), float(robot_field[i][1]))
                    bpos = [(float(p[0]), float(p[1])) for p in ball_field]

                    state.robot_field_positions = rpos
                    state.ball_field_positions = bpos
                    state.frame_positions[frame_idx] = {"robots": dict(rpos), "balls": list(bpos)}

                    tracked_balls = ball_dets
                    if len(ball_dets) > 0:
                        tracked_balls.tracker_id = np.arange(len(ball_dets))
                    state.shot_counter.update(
                        frame_index=frame_idx, timestamp=timestamp,
                        ball_detections=tracked_balls, ball_field_positions=ball_field,
                        robot_detections=tracked_robots, robot_field_positions=robot_field,
                        team_map=state.robot_tracker.team_map,
                    )

                    if tracked_robots.tracker_id is not None:
                        for i, tid in enumerate(tracked_robots.tracker_id):
                            label = state.robot_tracker.get_team(int(tid)) or f"unknown_{tid}"
                            robot_track_buffer.append({
                                "match_id": state.match_id, "team_label": label,
                                "frame": frame_idx,
                                "x": float(robot_field[i][0]), "y": float(robot_field[i][1]),
                                "ts": timestamp,
                            })
                    for p in ball_field:
                        ball_track_buffer.append({
                            "match_id": state.match_id, "frame": frame_idx,
                            "x": float(p[0]), "y": float(p[1]), "z": 0.0, "ts": timestamp,
                        })

                labels = []
                if tracked_robots.tracker_id is not None:
                    for tid in tracked_robots.tracker_id:
                        team = state.robot_tracker.get_team(int(tid))
                        labels.append(f"#{tid}" + (f" {team}" if team else ""))
                else:
                    labels = ["" for _ in range(len(tracked_robots))]

                annotated = frame.copy()
                if len(tracked_robots) > 0:
                    annotated = box_annotator.annotate(annotated, tracked_robots)
                    annotated = label_annotator.annotate(annotated, tracked_robots, labels=labels)
                if len(ball_dets) > 0:
                    annotated = box_annotator.annotate(annotated, ball_dets)

                if detect_count % FLUSH_INTERVAL == 0:
                    state.db.insert_robot_tracks(robot_track_buffer)
                    state.db.insert_ball_tracks(ball_track_buffer)
                    robot_track_buffer.clear()
                    ball_track_buffer.clear()

                elapsed = time.perf_counter() - t0
                fps_text.value = f"检测 FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else ""

            except Exception:
                logger.exception("Error processing frame {}", frame_idx)

        writer.write(annotated)

        now = time.perf_counter()
        if now - last_ui_update >= 0.25:
            last_ui_update = now
            pct = (frame_idx - start) / total_to_process
            progress_bar.value = pct
            progress_text.value = f"{pct * 100:.1f}%  帧 {frame_idx}/{end}"
            frame_text.value = f"帧: {frame_idx}"

            if run_detect:
                live_image.src = frame_to_bytes(annotated)
                update_field_overlay(state, field_stack)
                counts = state.shot_counter.counts
                shot_table.rows = [
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(team)),
                        ft.DataCell(ft.Text(str(cnt))),
                    ])
                    for team, cnt in sorted(counts.items())
                ]

            try:
                page.update()
            except Exception:
                pass

    # ── finalize ─────────────────────────────────────────────
    writer.release()
    cv_source.release()

    state.db.insert_robot_tracks(robot_track_buffer)
    state.db.insert_ball_tracks(ball_track_buffer)
    shot_rows = [
        {
            "match_id": state.match_id, "team_label": evt.team_label,
            "frame": evt.frame_index, "ts": evt.timestamp,
            "ball_x": evt.ball_position[0], "ball_y": evt.ball_position[1],
            "scored": evt.scored,
        }
        for evt in state.shot_counter.events
    ]
    state.db.insert_shot_events(shot_rows)

    state.running = False
    progress_bar.value = 1.0
    progress_text.value = "处理完成！"
    process_btn.text = "开始处理"
    process_btn.icon = ft.Icons.PLAY_ARROW
    process_btn.bgcolor = ft.Colors.GREEN_700
    status_text.value = "处理完成 — 播放标注视频"
    status_text.color = ft.Colors.GREEN
    fps_text.value = ""

    counts = state.shot_counter.counts
    shot_table.rows = [
        ft.DataRow(cells=[
            ft.DataCell(ft.Text(team)),
            ft.DataCell(ft.Text(str(cnt))),
        ])
        for team, cnt in sorted(counts.items())
    ]

    video_container.content = make_player([ftv.VideoMedia(output_path)])
    try:
        page.update()
    except Exception:
        pass
    logger.info("Batch processing complete: {} ({} frames written)", output_path, frames_read)
