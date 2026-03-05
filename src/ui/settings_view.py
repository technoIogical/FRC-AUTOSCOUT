"""Settings view: video selection, trimming, field calibration, scoring zones."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import flet as ft
import flet_video as ftv
import numpy as np
from loguru import logger

from config import FIELD_IMAGE_PATH, FIELD_LENGTH_M, FIELD_WIDTH_M
from src.shot_counter import ScoringZone
from src.video_source import VideoSource
from src.ui.shared import (
    AppState,
    VIDEO_W, VIDEO_H, MAP_W, MAP_H,
    _DRAG_RADIUS_PX,
    frame_to_bytes, fmt_time, draw_calib_overlay,
    make_player, update_field_overlay,
)


def build_settings(page: ft.Page, state: AppState, on_start_analysis) -> ft.Column:
    """Return the settings panel.

    *on_start_analysis* is a callback invoked when the user clicks
    "开始分析" to switch to the analysis view.
    """

    _ui = {"zoom": 1.0, "drag_idx": -1, "last_draw": 0.0}
    _THROTTLE_MS = 0.05  # 50 ms min interval between redraws during drag

    # ── Helpers ──────────────────────────────────────────────

    def _disp_to_pixel(dx: float, dy: float) -> tuple[float, float]:
        if state.start_frame_img is None:
            return (0.0, 0.0)
        z = _ui["zoom"]
        ih, iw = state.start_frame_img.shape[:2]
        return (dx / z * iw / VIDEO_W, dy / z * ih / VIDEO_H)

    def _refresh_calib():
        if state.start_frame_img is None:
            return
        z = _ui["zoom"]
        target_w = int(VIDEO_W * z)
        target_h = int(VIDEO_H * z)
        ih, iw = state.start_frame_img.shape[:2]
        if iw > target_w or ih > target_h:
            small = cv2.resize(state.start_frame_img, (target_w, target_h))
            scale = target_w / iw
            scaled_pts = [(px * scale, py * scale) for px, py in state.calibration_pixels]
            vis = draw_calib_overlay(small, scaled_pts)
        else:
            vis = draw_calib_overlay(state.start_frame_img, state.calibration_pixels)
            if z != 1.0:
                vis = cv2.resize(vis, (target_w, target_h))
        calib_img.src = frame_to_bytes(vis, quality=70)
        calib_img.width = target_w
        calib_img.height = target_h
        zoom_label.value = f"{int(z * 100)}%"
        confirm_btn.visible = (len(state.calibration_pixels) == 4
                               and state.calibration_mode)

    # ── Logging ──────────────────────────────────────────────

    log_list = ft.ListView(height=100, spacing=2, auto_scroll=True)

    def add_log(msg: str):
        log_list.controls.append(ft.Text(msg, size=11, color=ft.Colors.WHITE70))
        if len(log_list.controls) > 200:
            log_list.controls.pop(0)

    # ── Step 1: Video selection ──────────────────────────────

    video_player_container = ft.Container(
        content=make_player(), width=VIDEO_W, height=VIDEO_H + 40,
    )

    video_name_text = ft.Text("未选择视频", size=13, color=ft.Colors.WHITE54)
    match_id_field = ft.TextField(label="比赛 ID", value="match_001", width=180, dense=True)
    event_field = ft.TextField(label="赛事名称", value="", width=180, dense=True)

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

            video_player_container.content = make_player([ftv.VideoMedia(path)])
            video_name_text.value = f"已加载: {Path(path).name}"
            video_name_text.color = ft.Colors.BLUE
            start_label.value = "起点: 未设置"
            end_label.value = "终点: 未设置"
            calib_status.value = "标定: 未完成"
            calib_status.color = ft.Colors.WHITE54
            calib_btn.text = "场地标定"
            calib_btn.disabled = True
            zone_btn.disabled = True
            go_btn.disabled = True
            add_log(f"视频加载: {path}")
        except Exception as ex:
            video_name_text.value = f"加载失败: {ex}"
            video_name_text.color = ft.Colors.RED
        page.update()

    # ── Step 2: Trim — set start / end ───────────────────────

    start_label = ft.Text("起点: 未设置", size=12, color=ft.Colors.WHITE54)
    end_label = ft.Text("终点: 未设置", size=12, color=ft.Colors.WHITE54)

    async def set_start(e):
        if state.source is None:
            add_log("请先加载视频"); page.update(); return
        pos_secs = 0.0
        try:
            pos = await video_player_container.content.get_current_position()
            pos_secs = pos.in_milliseconds / 1000.0
        except Exception:
            pass
        fps = state.source.meta.fps or 30.0
        state.start_frame = int(pos_secs * fps)
        start_label.value = f"起点: {fmt_time(pos_secs)} (帧 {state.start_frame})"
        start_label.color = ft.Colors.GREEN

        state.source.seek(state.start_frame)
        ok, frame = state.source.read_frame()
        if ok and frame is not None:
            state.start_frame_img = frame
            state.calibration_mode = False
            state.calibration_pixels = []
            _ui["zoom"] = 1.0
            _refresh_calib()
            calib_section_container.content = calib_view
            calib_btn.disabled = False
            zone_btn.disabled = False
            go_btn.disabled = False

        add_log(f"起点 {fmt_time(pos_secs)}")
        page.update()

    async def set_end(e):
        if state.source is None:
            add_log("请先加载视频"); page.update(); return
        pos_secs = 0.0
        try:
            pos = await video_player_container.content.get_current_position()
            pos_secs = pos.in_milliseconds / 1000.0
        except Exception:
            pass
        fps = state.source.meta.fps or 30.0
        state.end_frame = int(pos_secs * fps)
        end_label.value = f"终点: {fmt_time(pos_secs)} (帧 {state.end_frame})"
        end_label.color = ft.Colors.GREEN
        add_log(f"终点 {fmt_time(pos_secs)}")
        page.update()

    # ── Step 3: Field calibration ────────────────────────────

    calib_img = ft.Image(src=b"", width=VIDEO_W, height=VIDEO_H, fit="fill", border_radius=4)
    zoom_label = ft.Text("100%", size=12)
    confirm_btn = ft.Button(
        "确认标定", icon=ft.Icons.CHECK,
        bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE,
        visible=False,
    )
    calib_status = ft.Text("标定: 未完成", size=12, color=ft.Colors.WHITE54)

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
        now = time.perf_counter()
        if now - _ui["last_draw"] < _THROTTLE_MS:
            return
        _ui["last_draw"] = now
        _refresh_calib()
        page.update()

    def _calib_pan_end(e):
        if _ui["drag_idx"] >= 0:
            idx = _ui["drag_idx"]
            if idx < len(state.calibration_pixels):
                px, py = state.calibration_pixels[idx]
                add_log(f"标定点 {idx + 1} → ({px:.0f}, {py:.0f})")
            _ui["drag_idx"] = -1
            _refresh_calib()
            page.update()

    calib_gesture = ft.GestureDetector(
        content=calib_img,
        on_tap_up=_calib_tap,
        on_pan_down=_calib_pan_down,
        on_pan_update=_calib_pan_update,
        on_pan_end=_calib_pan_end,
    )

    def _zoom_in(e):
        _ui["zoom"] = min(3.0, _ui["zoom"] + 0.25); _refresh_calib(); page.update()

    def _zoom_out(e):
        _ui["zoom"] = max(0.5, _ui["zoom"] - 0.25); _refresh_calib(); page.update()

    def _zoom_reset(e):
        _ui["zoom"] = 1.0; _refresh_calib(); page.update()

    def _confirm_calib(e):
        if len(state.calibration_pixels) != 4:
            return
        corners = np.array(state.calibration_pixels, dtype=np.float32)
        state.robot_tracker.homography.calibrate(corners)
        state.calibration_mode = False
        confirm_btn.visible = False
        calib_status.value = "标定: 已完成 ✓"
        calib_status.color = ft.Colors.GREEN
        calib_btn.text = "重新标定"
        add_log("场地标定完成！")
        _refresh_calib()
        update_field_overlay(state, field_stack)
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
            ft.Column(controls=[calib_gesture], scroll=ft.ScrollMode.AUTO, height=VIDEO_H)
        ],
        scroll=ft.ScrollMode.AUTO, width=VIDEO_W, height=VIDEO_H,
    )

    calib_view = ft.Column(controls=[calib_toolbar, calib_scroll], spacing=2)
    calib_section_container = ft.Container(
        content=ft.Text("请先设为起点以抓取标定帧", size=13, color=ft.Colors.WHITE54),
        width=VIDEO_W, height=VIDEO_H + 40,
    )

    calib_btn = ft.Button("场地标定", icon=ft.Icons.CROP_FREE, disabled=True)

    def toggle_calibration(e):
        if state.start_frame_img is None:
            add_log("请先设为起点"); page.update(); return
        state.calibration_mode = True
        state.calibration_pixels = []
        _ui["zoom"] = 1.0
        _refresh_calib()
        calib_section_container.content = calib_view
        add_log("标定模式：在画面上依次点击场地四角（左上→右上→右下→左下），可拖动调整")
        page.update()

    calib_btn.on_click = toggle_calibration

    # ── Step 4: Scoring zones (on field map) ─────────────────

    field_canvas = ft.Image(
        src=str(FIELD_IMAGE_PATH) if FIELD_IMAGE_PATH.exists() else "",
        width=MAP_W, height=MAP_H, fit="fill", border_radius=8,
    )
    field_stack = ft.Stack(controls=[field_canvas], width=MAP_W, height=MAP_H)

    zone_btn = ft.Button("画得分区", icon=ft.Icons.RECTANGLE_OUTLINED, disabled=True)

    def toggle_zone_draw(e):
        state.zone_draw_mode = not state.zone_draw_mode
        state.zone_start = None
        if state.zone_draw_mode:
            add_log("画框模式：在场地图上点击两个对角点定义得分区")
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
            update_field_overlay(state, field_stack)
        page.update()

    field_gesture = ft.GestureDetector(content=field_stack, on_tap_up=on_field_click)

    # ── "开始分析" button ────────────────────────────────────

    go_btn = ft.Button(
        "开始分析 →", icon=ft.Icons.ROCKET_LAUNCH,
        bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE,
        disabled=True,
        height=48,
    )

    def _go(e):
        if not state.is_ready:
            add_log("请先完成视频选择和起点设置")
            page.update()
            return
        state.match_id = match_id_field.value or "match_001"
        on_start_analysis()

    go_btn.on_click = _go

    # ── Layout ───────────────────────────────────────────────

    step1 = ft.Container(
        content=ft.Column([
            ft.Text("① 选择视频", size=15, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.Button("打开视频", icon=ft.Icons.VIDEO_FILE, on_click=open_file),
                video_name_text,
            ], spacing=8),
            ft.Row([match_id_field, event_field], spacing=8),
            video_player_container,
        ], spacing=6),
        padding=10,
        border=ft.border.all(1, ft.Colors.WHITE24),
        border_radius=8,
    )

    step2 = ft.Container(
        content=ft.Column([
            ft.Text("② 裁切视频", size=15, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.Button("设为起点", icon=ft.Icons.FLAG, on_click=set_start),
                ft.Button("设为终点", icon=ft.Icons.SPORTS_SCORE, on_click=set_end),
                ft.VerticalDivider(),
                start_label, end_label,
            ], spacing=8),
        ], spacing=6),
        padding=10,
        border=ft.border.all(1, ft.Colors.WHITE24),
        border_radius=8,
    )

    step3 = ft.Container(
        content=ft.Column([
            ft.Text("③ 场地标定", size=15, weight=ft.FontWeight.BOLD),
            ft.Row([calib_btn, calib_status], spacing=8),
            calib_section_container,
        ], spacing=6),
        padding=10,
        border=ft.border.all(1, ft.Colors.WHITE24),
        border_radius=8,
    )

    step4 = ft.Container(
        content=ft.Column([
            ft.Text("④ 得分区", size=15, weight=ft.FontWeight.BOLD),
            ft.Row([zone_btn], spacing=8),
            field_gesture,
        ], spacing=6),
        padding=10,
        border=ft.border.all(1, ft.Colors.WHITE24),
        border_radius=8,
    )

    left_col = ft.Column(
        controls=[step1, step2, step3],
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    right_col = ft.Column(
        controls=[
            step4,
            ft.Container(expand=True),
            go_btn,
        ],
        spacing=8,
        width=MAP_W + 30,
    )

    main_row = ft.Row(
        controls=[left_col, ft.VerticalDivider(), right_col],
        spacing=12,
        expand=True,
    )

    log_section = ft.Container(
        content=ft.Column([
            ft.Text("日志", size=12, weight=ft.FontWeight.BOLD),
            ft.Container(content=log_list, bgcolor=ft.Colors.BLACK87,
                         border_radius=6, padding=6, height=100),
        ], spacing=2),
    )

    return ft.Column(
        controls=[main_row, ft.Divider(), log_section],
        spacing=6,
        expand=True,
    )
