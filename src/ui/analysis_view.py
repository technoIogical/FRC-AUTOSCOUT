"""Analysis view: real-time detection, field map, shot stats, playback."""
from __future__ import annotations

import threading
from pathlib import Path

import flet as ft
import flet_video as ftv
import numpy as np
from loguru import logger

from config import (
    BASE_DIR,
    FIELD_IMAGE_PATH,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
)
from src.detector import RobotBallDetector
from src.video_source import VideoSource
from src.ui.shared import (
    AppState,
    VIDEO_W, VIDEO_H, MAP_W, MAP_H,
    frame_to_bytes, make_player,
    update_field_overlay, batch_process,
)


def build_analysis(page: ft.Page, state: AppState) -> ft.Column:
    """Return the analysis panel."""

    # ── UI elements ──────────────────────────────────────────

    status_text = ft.Text("就绪", size=14, color=ft.Colors.GREEN)
    fps_text = ft.Text("FPS: --", size=12)
    frame_text = ft.Text("帧: 0", size=12)

    live_image = ft.Image(src=b"", width=VIDEO_W, height=VIDEO_H,
                          fit="contain", border_radius=8, visible=False)
    placeholder = ft.Container(
        content=ft.Text("在设置页完成配置后点击「开始处理」", size=14, color=ft.Colors.WHITE54,
                        text_align=ft.TextAlign.CENTER),
        width=VIDEO_W, height=VIDEO_H,
        bgcolor=ft.Colors.BLACK,
        alignment=ft.Alignment(0, 0),
        border_radius=8,
    )
    video_container = ft.Container(
        content=ft.Stack(controls=[placeholder, live_image]),
        width=VIDEO_W, height=VIDEO_H + 40,
    )

    progress_bar = ft.ProgressBar(width=VIDEO_W, value=0, visible=False)
    progress_text = ft.Text("", size=12, visible=False)

    field_canvas = ft.Image(
        src=str(FIELD_IMAGE_PATH) if FIELD_IMAGE_PATH.exists() else "",
        width=MAP_W, height=MAP_H, fit="fill", border_radius=8,
    )
    field_stack = ft.Stack(controls=[field_canvas], width=MAP_W, height=MAP_H)

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

    process_btn = ft.Button(
        "开始处理", icon=ft.Icons.PLAY_ARROW,
        bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE,
    )

    def start_processing(e):
        if state.running:
            state.running = False
            return

        if not state.is_ready:
            add_log("请先在设置页完成视频选择和起点设置")
            page.update()
            return

        if state.detector is None:
            add_log("正在加载模型…"); page.update()
            state.detector = RobotBallDetector()
            add_log("模型加载完成")

        if not state.match_id:
            state.match_id = "match_001"
        state.db.create_match(state.match_id, "")
        state.running = True
        state.frame_positions.clear()

        process_btn.text = "停止处理"
        process_btn.icon = ft.Icons.STOP
        process_btn.bgcolor = ft.Colors.RED_700
        progress_bar.visible = True
        progress_bar.value = 0
        progress_text.visible = True
        progress_text.value = "0%"
        status_text.value = "处理中…"
        status_text.color = ft.Colors.AMBER

        if state.start_frame_img is not None:
            live_image.src = frame_to_bytes(state.start_frame_img)
        live_image.visible = True
        placeholder.visible = False
        page.update()

        cv_source = VideoSource.from_file(state.video_path)
        output_path = str(BASE_DIR / "data" / f"annotated_{state.match_id}.mp4")

        t = threading.Thread(
            target=batch_process,
            args=(state, page, cv_source, output_path,
                  live_image, field_stack,
                  fps_text, frame_text, shot_table,
                  progress_bar, progress_text,
                  process_btn, status_text,
                  video_container, add_log),
            daemon=True,
        )
        t.start()

    process_btn.on_click = start_processing

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

    toolbar = ft.Row(
        controls=[
            process_btn,
            ft.VerticalDivider(),
            status_text, ft.VerticalDivider(), fps_text,
            ft.VerticalDivider(), frame_text,
        ],
        spacing=8,
    )

    assign_row = ft.Row(
        controls=[tracker_id_field, team_assign_dropdown, assign_btn],
        spacing=8,
    )

    left_panel = ft.Column(
        controls=[
            ft.Text("检测结果", size=16, weight=ft.FontWeight.BOLD),
            video_container,
            progress_bar,
            progress_text,
        ],
        spacing=4, width=VIDEO_W + 20,
    )

    right_panel = ft.Column(
        controls=[
            ft.Text("场地地图", size=16, weight=ft.FontWeight.BOLD),
            field_stack,
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
                ft.Container(expand=True),
                ft.Button("同步云端", icon=ft.Icons.CLOUD_UPLOAD, on_click=sync_cloud),
                ft.Button("生成报告", icon=ft.Icons.ANALYTICS, on_click=generate_report),
            ], spacing=8),
            ft.Text("日志", size=12, weight=ft.FontWeight.BOLD),
            ft.Container(content=log_list, bgcolor=ft.Colors.BLACK87,
                         border_radius=6, padding=6, height=130),
        ],
        spacing=4,
    )

    return ft.Column(
        controls=[toolbar, ft.Divider(), main_area, ft.Divider(), bottom_bar],
        spacing=6, expand=True,
    )
