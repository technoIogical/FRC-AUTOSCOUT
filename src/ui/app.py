"""Main application entry: NavigationRail with Settings and Analysis views."""
from __future__ import annotations

import flet as ft
from loguru import logger

from src.ui.shared import AppState
from src.ui.settings_view import build_settings
from src.ui.analysis_view import build_analysis


def build_app(page: ft.Page) -> None:
    page.title = "FRC AutoScout"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.window.width = 1400
    page.window.height = 900

    state = AppState()

    content_area = ft.Container(expand=True, padding=10)

    def switch_to_analysis():
        rail.selected_index = 1
        content_area.content = analysis_panel
        page.update()

    settings_panel = build_settings(page, state, on_start_analysis=switch_to_analysis)
    analysis_panel = build_analysis(page, state)

    content_area.content = settings_panel

    def on_nav_change(e):
        idx = e.control.selected_index
        if idx == 0:
            content_area.content = settings_panel
        else:
            content_area.content = analysis_panel
        page.update()

    rail = ft.NavigationRail(
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.SETTINGS, label="设置"),
            ft.NavigationRailDestination(icon=ft.Icons.ANALYTICS, label="分析"),
        ],
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=72,
        on_change=on_nav_change,
    )

    page.add(
        ft.Row(
            controls=[rail, ft.VerticalDivider(width=1), content_area],
            expand=True,
            spacing=0,
        )
    )
