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


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _draw_calib_overlay(frame: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """Draw calibration markers and connecting lines on a copy of frame."""
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


class AppState:
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


def build_ui(page: ft.Page) -> None:
    page.title = "FRC AutoScout"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10
    page.window.width = 1400
    page.window.height = 900

    state = AppState()
    _ui = {"zoom": 1.0, "drag_idx": -1}

    # ── helpers ───────────────────────────────────────────────

    def _make_player(playlist=None):
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

    def _fmt_time(secs: float) -> str:
        m, s = divmod(int(secs), 60)
        return f"{m}:{s:02d}"

    def _disp_to_pixel(dx: float, dy: float) -> tuple[float, float]:
        """Display coords (in zoomed image space) → original image pixels."""
        if state.start_frame_img is None:
            return (0.0, 0.0)
        z = _ui["zoom"]
        ih, iw = state.start_frame_img.shape[:2]
        return (dx / z * iw / VIDEO_W, dy / z * ih / VIDEO_H)

    def _pixel_to_disp(px: float, py: float) -> tuple[float, float]:
        """Original image pixels → zoomed display coords."""
        if state.start_frame_img is None:
            return (0.0, 0.0)
        z = _ui["zoom"]
        ih, iw = state.start_frame_img.shape[:2]
        return (px * VIDEO_W / iw * z, py * VIDEO_H / ih * z)

    def _refresh_calib():
        """Redraw calibration overlay with current points and zoom."""
        if state.start_frame_img is None:
            return
        vis = _draw_calib_overlay(state.start_frame_img, state.calibration_pixels)
        z = _ui["zoom"]
        if z != 1.0:
            h, w = vis.shape[:2]
            vis = cv2.resize(vis, (int(w * z), int(h * z)))
        calib_img.src = _frame_to_bytes(vis)
        calib_img.width = int(VIDEO_W * z)
        calib_img.height = int(VIDEO_H * z)
        zoom_label.value = f"{int(z * 100)}%"
        confirm_btn.visible = (len(state.calibration_pixels) == 4
                               and state.calibration_mode)

    # ── Calibration view widgets ─────────────────────────────

    calib_img = ft.Image(src=b"", width=VIDEO_W, height=VIDEO_H, fit="fill", border_radius=4)
    zoom_label = ft.Text("100%", size=12)
    confirm_btn = ft.Button(
        "确认标定", icon=ft.Icons.CHECK,
        bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE,
        visible=False,
    )

    def _calib_tap(e: ft.TapEvent):
        if not state.calibration_mode or state.start_frame_img is None:
            return
        if len(state.calibration_pixels) >= 4:
            add_log("已有4个标定点，拖动调整或点击「确认标定」")
            page.update()
            return
        px, py = _disp_to_pixel(e.local_position.x, e.local_position.y)
        state.calibration_pixels.append((px, py))
        add_log(f"标定点 {len(state.calibration_pixels)}/4: ({px:.0f}, {py:.0f})")
        _refresh_calib()
        page.update()

    def _calib_pan_down(e):
        """Detect if the press is near an existing point → start drag."""
        if not state.calibration_mode or state.start_frame_img is None:
            return
        px, py = _disp_to_pixel(e.local_position.x, e.local_position.y)
        for i, (cx, cy) in enumerate(state.calibration_pixels):
            if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 < _DRAG_RADIUS_PX:
                _ui["drag_idx"] = i
                return

    def _calib_pan_update(e):
        idx = _ui["drag_idx"]
        if idx < 0 or not state.calibration_mode or state.start_frame_img is None:
            return
        ih, iw = state.start_frame_img.shape[:2]
        px, py = _disp_to_pixel(e.local_position.x, e.local_position.y)
        px = max(0.0, min(float(iw), px))
        py = max(0.0, min(float(ih), py))
        state.calibration_pixels[idx] = (px, py)
        _refresh_calib()
        page.update()

    def _calib_pan_end(e):
        if _ui["drag_idx"] >= 0:
            idx = _ui["drag_idx"]
            if idx < len(state.calibration_pixels):
                px, py = state.calibration_pixels[idx]
                add_log(f"标定点 {idx + 1} → ({px:.0f}, {py:.0f})")
            _ui["drag_idx"] = -1
            page.update()

    calib_gesture = ft.GestureDetector(
        content=calib_img,
        on_tap_up=_calib_tap,
        on_pan_down=_calib_pan_down,
        on_pan_update=_calib_pan_update,
        on_pan_end=_calib_pan_end,
    )

    def _zoom_in(e):
        _ui["zoom"] = min(3.0, _ui["zoom"] + 0.25)
        _refresh_calib()
        page.update()

    def _zoom_out(e):
        _ui["zoom"] = max(0.5, _ui["zoom"] - 0.25)
        _refresh_calib()
        page.update()

    def _zoom_reset(e):
        _ui["zoom"] = 1.0
        _refresh_calib()
        page.update()

    def _confirm_calib(e):
        if len(state.calibration_pixels) != 4:
            return
        corners = np.array(state.calibration_pixels, dtype=np.float32)
        state.robot_tracker.homography.calibrate(corners)
        state.calibration_mode = False
        confirm_btn.visible = False
        calib_label.value = "标定: 已完成 ✓"
        calib_label.color = ft.Colors.GREEN
        calib_btn_main.text = "重新标定"
        add_log("场地标定完成！")
        _refresh_calib()
        _update_field_overlay(state, field_stack)
        page.update()

    confirm_btn.on_click = _confirm_calib

    calib_toolbar = ft.Row(
        controls=[
            ft.IconButton(ft.Icons.ZOOM_OUT, on_click=_zoom_out, tooltip="缩小"),
            zoom_label,
            ft.IconButton(ft.Icons.ZOOM_IN, on_click=_zoom_in, tooltip="放大"),
            ft.IconButton(ft.Icons.ZOOM_OUT_MAP, on_click=_zoom_reset, tooltip="重置"),
            ft.Container(expand=True),
            confirm_btn,
        ],
        spacing=4,
    )

    calib_scroll = ft.Row(
        controls=[
            ft.Column(
                controls=[calib_gesture],
                scroll=ft.ScrollMode.AUTO,
                height=VIDEO_H,
            )
        ],
        scroll=ft.ScrollMode.AUTO,
        width=VIDEO_W,
        height=VIDEO_H,
    )

    calib_view = ft.Column(controls=[calib_toolbar, calib_scroll], spacing=2)

    # ── Main UI elements ─────────────────────────────────────

    video_player = _make_player()
    video_container = ft.Container(content=video_player, width=VIDEO_W, height=VIDEO_H + 40)

    field_canvas = ft.Image(
        src=str(FIELD_IMAGE_PATH) if FIELD_IMAGE_PATH.exists() else "",
        width=MAP_W, height=MAP_H, fit="fill", border_radius=8,
    )
    field_stack = ft.Stack(controls=[field_canvas], width=MAP_W, height=MAP_H)

    status_text = ft.Text("就绪", size=14, color=ft.Colors.GREEN)
    fps_text = ft.Text("FPS: --", size=12)
    frame_text = ft.Text("帧: 0", size=12)
    start_label = ft.Text("起点: 未设置", size=12, color=ft.Colors.WHITE54)
    calib_label = ft.Text("标定: 未完成", size=12, color=ft.Colors.WHITE54)

    match_id_field = ft.TextField(label="比赛 ID", value="match_001", width=180, dense=True)
    event_field = ft.TextField(label="赛事名称", value="", width=180, dense=True)

    progress_bar = ft.ProgressBar(width=VIDEO_W, value=0, visible=False)
    progress_text = ft.Text("", size=12, visible=False)

    shot_table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text("队伍")), ft.DataColumn(ft.Text("投球数"))],
        rows=[], width=300,
    )

    team_assign_dropdown = ft.Dropdown(
        label="指定队伍", width=160, dense=True,
        options=[
            ft.dropdown.Option("Red1"), ft.dropdown.Option("Red2"), ft.dropdown.Option("Red3"),
            ft.dropdown.Option("Blue1"), ft.dropdown.Option("Blue2"), ft.dropdown.Option("Blue3"),
        ],
    )
    tracker_id_field = ft.TextField(label="Tracker ID", width=80, dense=True)
    assign_btn = ft.Button("指定")

    log_list = ft.ListView(height=120, spacing=2, auto_scroll=True)

    def add_log(msg: str):
        log_list.controls.append(ft.Text(msg, size=11, color=ft.Colors.WHITE70))
        if len(log_list.controls) > 200:
            log_list.controls.pop(0)

    calib_btn_main = ft.Button("场地标定", icon=ft.Icons.CROP_FREE, disabled=True)
    zone_btn = ft.Button("画得分区", icon=ft.Icons.RECTANGLE_OUTLINED, disabled=True)
    process_btn = ft.Button(
        "开始处理", icon=ft.Icons.PLAY_ARROW,
        bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE,
        disabled=True,
    )

    # ── Callbacks ────────────────────────────────────────────

    def on_assign(e):
        try:
            tid = int(tracker_id_field.value)
        except (ValueError, TypeError):
            add_log("无效的 Tracker ID"); page.update(); return
        label = team_assign_dropdown.value
        if not label:
            add_log("请选择队伍"); page.update(); return
        state.robot_tracker.assign_team(tid, label)
        add_log(f"Tracker {tid} → {label}")
        page.update()

    assign_btn.on_click = on_assign

    # ── Open video ───────────────────────────────────────────

    async def open_file(e):
        files = await ft.FilePicker().pick_files(
            dialog_title="选择视频文件",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4", "avi", "mov", "mkv"],
        )
        if not files:
            return
        path = files[0].path
        state.video_path = path
        try:
            state.source = VideoSource.from_file(path)
            total = state.source.meta.total_frames or 0
            state.total_frames = total
            state.start_frame = 0
            state.end_frame = total
            state.start_frame_img = None
            state.calibration_pixels = []
            state.calibration_mode = False

            video_container.content = _make_player([ftv.VideoMedia(path)])
            calib_btn_main.disabled = True
            calib_btn_main.text = "场地标定"
            zone_btn.disabled = True
            process_btn.disabled = True
            start_label.value = "起点: 未设置"
            calib_label.value = "标定: 未完成"
            calib_label.color = ft.Colors.WHITE54
            status_text.value = f"已加载: {Path(path).name}"
            status_text.color = ft.Colors.BLUE
            add_log(f"视频加载: {path}  →  浏览后点击「设为起点」")
        except Exception as ex:
            status_text.value = f"加载失败: {ex}"
            status_text.color = ft.Colors.RED
        page.update()

    # ── Set start → grab first frame → show calibration view ─

    async def set_start(e):
        if state.source is None:
            add_log("请先加载视频"); page.update(); return

        pos_secs = 0.0
        try:
            pos = await video_container.content.get_current_position()
            pos_secs = pos.in_milliseconds / 1000.0
        except Exception:
            pass

        fps = state.source.meta.fps or 30.0
        state.start_frame = int(pos_secs * fps)
        start_label.value = f"起点: {_fmt_time(pos_secs)} (帧 {state.start_frame})"

        state.source.seek(state.start_frame)
        ok, frame = state.source.read_frame()
        if not ok or frame is None:
            add_log("抓帧失败"); page.update(); return

        state.start_frame_img = frame
        state.calibration_mode = False
        state.calibration_pixels = []
        _ui["zoom"] = 1.0
        _refresh_calib()

        video_container.content = calib_view
        calib_btn_main.disabled = False
        zone_btn.disabled = False
        process_btn.disabled = False
        add_log(f"起点 {_fmt_time(pos_secs)}  →  点击「场地标定」在画面上标定四角")
        page.update()

    # ── Toggle calibration mode ──────────────────────────────

    def toggle_calibration(e):
        if state.start_frame_img is None:
            add_log("请先设为起点"); page.update(); return
        state.calibration_mode = True
        state.calibration_pixels = []
        _ui["zoom"] = 1.0
        _refresh_calib()
        video_container.content = calib_view
        add_log("标定模式：在画面上依次点击场地四角（左上→右上→右下→左下），可拖动调整")
        page.update()

    calib_btn_main.on_click = toggle_calibration

    # ── Zone drawing (on field map) ──────────────────────────

    def toggle_zone_draw(e):
        state.zone_draw_mode = not state.zone_draw_mode
        state.zone_start = None
        if state.zone_draw_mode:
            add_log("画框模式：在右侧场地图上点击两个对角点定义得分区")
        else:
            add_log("画框模式已关闭")
        page.update()

    zone_btn.on_click = toggle_zone_draw

    def on_field_click(e: ft.TapEvent):
        if not state.zone_draw_mode:
            return
        fx = e.local_position.x / MAP_W * FIELD_LENGTH_M
        fy = e.local_position.y / MAP_H * FIELD_WIDTH_M
        if state.zone_start is None:
            state.zone_start = (fx, fy)
            add_log(f"得分区起点: ({fx:.2f}, {fy:.2f})")
        else:
            x1, y1 = state.zone_start
            zone = ScoringZone(
                x_min=min(x1, fx), y_min=min(y1, fy),
                x_max=max(x1, fx), y_max=max(y1, fy),
                name=f"Zone{len(state.shot_counter._zones) + 1}",
            )
            state.shot_counter.add_zone(zone)
            state.zone_start = None
            state.zone_draw_mode = False
            add_log(f"得分区已添加: {zone.name}")
            _update_field_overlay(state, field_stack)
        page.update()

    field_gesture = ft.GestureDetector(content=field_stack, on_tap_up=on_field_click)

    # ── Batch processing ─────────────────────────────────────

    async def start_processing(e):
        if state.running:
            state.running = False
            return

        if state.source is None or not state.video_path:
            add_log("请先加载视频并设为起点"); page.update(); return

        if state.detector is None:
            add_log("正在加载模型…"); page.update()
            state.detector = RobotBallDetector()
            add_log("模型加载完成")

        state.match_id = match_id_field.value or "match_001"
        state.db.create_match(state.match_id, event_field.value)
        state.running = True
        state.frame_positions.clear()

        process_btn.text = "停止处理"
        process_btn.icon = ft.Icons.STOP
        process_btn.bgcolor = ft.Colors.RED_700
        calib_btn_main.disabled = True
        zone_btn.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        progress_text.visible = True
        progress_text.value = "0%"
        status_text.value = "处理中…"
        status_text.color = ft.Colors.AMBER

        live_image = ft.Image(
            src=_frame_to_bytes(state.start_frame_img) if state.start_frame_img is not None else b"",
            width=VIDEO_W, height=VIDEO_H,
            fit="contain", border_radius=8,
        )
        video_container.content = live_image
        page.update()

        cv_source = VideoSource.from_file(state.video_path)
        output_path = str(BASE_DIR / "data" / f"annotated_{state.match_id}.mp4")

        t = threading.Thread(
            target=_batch_process,
            args=(state, page, cv_source, output_path,
                  live_image, field_stack,
                  fps_text, frame_text, shot_table,
                  progress_bar, progress_text,
                  process_btn, calib_btn_main, zone_btn,
                  status_text, video_container, _make_player, add_log),
            daemon=True,
        )
        t.start()

    process_btn.on_click = start_processing

    # ── Cloud / report ───────────────────────────────────────

    def sync_cloud(e):
        if not state.cloud_db.is_configured:
            add_log("Supabase 未配置"); page.update(); return
        try:
            state.cloud_db.sync_all_from_local(state.db)
            add_log("云端同步完成")
        except Exception as ex:
            add_log(f"同步失败: {ex}")
        page.update()

    def generate_report(e):
        if not state.match_id:
            add_log("没有活跃比赛"); page.update(); return
        try:
            p = state.analyzer.generate_profile_report(state.match_id)
            add_log(f"报告已生成: {p}")
        except Exception as ex:
            add_log(f"报告失败: {ex}")
        page.update()

    # ── Layout ───────────────────────────────────────────────

    toolbar_row1 = ft.Row(
        controls=[
            ft.Button("打开视频", icon=ft.Icons.VIDEO_FILE, on_click=open_file),
            ft.Button("设为起点", icon=ft.Icons.FLAG, on_click=set_start),
            ft.VerticalDivider(),
            match_id_field, event_field,
        ],
        spacing=8, scroll=ft.ScrollMode.AUTO,
    )

    toolbar_row2 = ft.Row(
        controls=[
            calib_btn_main, zone_btn,
            ft.VerticalDivider(),
            process_btn,
            ft.VerticalDivider(),
            start_label, calib_label,
        ],
        spacing=8,
    )

    assign_row = ft.Row(
        controls=[tracker_id_field, team_assign_dropdown, assign_btn],
        spacing=8,
    )

    left_panel = ft.Column(
        controls=[
            ft.Text("视频 / 检测结果", size=16, weight=ft.FontWeight.BOLD),
            video_container,
            progress_bar,
            progress_text,
        ],
        spacing=4, width=VIDEO_W + 20,
    )

    right_panel = ft.Column(
        controls=[
            ft.Text("场地地图", size=16, weight=ft.FontWeight.BOLD),
            field_gesture,
            assign_row,
            ft.Divider(),
            ft.Text("投球统计", size=14, weight=ft.FontWeight.BOLD),
            shot_table,
        ],
        spacing=8, width=520,
    )

    main_area = ft.Row(
        controls=[left_panel, ft.VerticalDivider(), right_panel],
        spacing=12, expand=True,
    )

    bottom_bar = ft.Column(
        controls=[
            ft.Row([
                status_text, ft.VerticalDivider(), fps_text,
                ft.VerticalDivider(), frame_text,
                ft.Container(expand=True),
                ft.Button("同步云端", icon=ft.Icons.CLOUD_UPLOAD, on_click=sync_cloud),
                ft.Button("生成报告", icon=ft.Icons.ANALYTICS, on_click=generate_report),
            ], spacing=8),
            ft.Text("日志", size=12, weight=ft.FontWeight.BOLD),
            ft.Container(content=log_list, bgcolor=ft.Colors.BLACK87, border_radius=6, padding=6, height=130),
        ],
        spacing=4,
    )

    page.add(
        ft.Column(
            controls=[toolbar_row1, toolbar_row2, ft.Divider(), main_area, ft.Divider(), bottom_bar],
            spacing=6, expand=True,
        )
    )


# ── Batch processing loop ────────────────────────────────────

def _batch_process(
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
    calib_btn: ft.Button,
    zone_btn: ft.Button,
    status_text: ft.Text,
    video_container: ft.Container,
    make_player,
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
    if not writer.isOpened():
        logger.error("Failed to open VideoWriter: {}", output_path)
        state.running = False
        return

    robot_track_buffer: list[dict] = []
    ball_track_buffer: list[dict] = []
    FLUSH_INTERVAL = 30
    detect_count = 0
    last_ui_update = 0.0

    logger.info("Batch processing: frames {}-{} → {}", start, end, output_path)

    for frame_idx in range(start, end):
        if not state.running:
            break

        cv_source.seek(frame_idx)
        ok, frame = cv_source.read_frame()
        if not ok or frame is None:
            continue

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

                # annotate
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
                live_image.src = _frame_to_bytes(annotated)
                _update_field_overlay(state, field_stack)
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
    calib_btn.disabled = False
    zone_btn.disabled = False
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
    logger.info("Batch processing complete: {}", output_path)


# ── Field overlay (zones + robots + balls) ───────────────────

def _update_field_overlay(state: AppState, field_stack: ft.Stack):
    field_stack.controls = [field_stack.controls[0]]

    # scoring zones
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
                bgcolor=ft.Colors.GREEN,
                opacity=0.35,
                border=ft.border.all(2, ft.Colors.LIGHT_GREEN),
                left=zl, top=zt,
                alignment=ft.Alignment(0, 0),
            )
        )

    # robots
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

    # balls
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
