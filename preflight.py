"""Pre-flight check — run before every delivery to catch issues early.

Usage:
    python preflight.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
errors = 0


def _assert_path(p: Path):
    if not p.exists():
        raise FileNotFoundError(str(p))


def assert_(cond, msg="assertion failed"):
    if not cond:
        raise AssertionError(msg)


def check(label: str, fn):
    global errors
    try:
        fn()
        print(f"  {PASS} {label}")
    except Exception as e:
        errors += 1
        print(f"  {FAIL} {label}: {e}")


def warn_check(label: str, fn):
    try:
        fn()
        print(f"  {PASS} {label}")
    except Exception as e:
        print(f"  {WARN} {label}: {e}")


# ── 1. Dependencies ─────────────────────────────────────────
print("\n[1/6] Dependencies")
for mod in ["cv2", "flet", "numpy", "supervision", "ultralytics",
            "duckdb", "loguru", "polars", "pandas"]:
    check(f"import {mod}", lambda m=mod: importlib.import_module(m))

# ── 2. Config & paths ───────────────────────────────────────
print("\n[2/6] Config & resource paths")
import config

check("config loads", lambda: None)
check("ROBOT_MODEL_PATH exists", lambda: _assert_path(config.ROBOT_MODEL_PATH))
check("BALL_MODEL_PATH exists", lambda: _assert_path(config.BALL_MODEL_PATH))
check("FIELD_IMAGE_PATH exists", lambda: _assert_path(config.FIELD_IMAGE_PATH))
check("ROBOT_RED_ICON exists", lambda: _assert_path(config.ROBOT_RED_ICON))
check("ROBOT_BLUE_ICON exists", lambda: _assert_path(config.ROBOT_BLUE_ICON))
check("data/ dir exists", lambda: _assert_path(config.DUCKDB_PATH.parent))

# ── 3. Module imports ────────────────────────────────────────
print("\n[3/6] Project module imports")
modules = [
    "src.video_source",
    "src.detector",
    "src.tracker",
    "src.shot_counter",
    "src.db.local_db",
    "src.db.cloud_db",
    "src.analysis.analyzer",
    "src.ui.main_view",
]
for mod in modules:
    check(f"import {mod}", lambda m=mod: importlib.import_module(m))

# ── 4. Flet API compatibility ───────────────────────────────
print("\n[4/6] Flet API compatibility")
import flet as ft

def _check_no_removed_attrs():
    removed = []
    for attr in ["ImageFit"]:
        if not hasattr(ft, attr):
            pass  # expected removal, ok
        else:
            removed.append(attr)

check("ft.FilePicker exists", lambda: assert_(hasattr(ft, "FilePicker")))
check("ft.Image accepts bytes src", lambda: ft.Image(src=b"\xff"))
check("ft.Colors.GREEN exists", lambda: assert_(hasattr(ft.Colors, "GREEN")))
check("ft.Icons.PLAY_ARROW exists", lambda: assert_(hasattr(ft.Icons, "PLAY_ARROW")))
check("ft.ThemeMode.DARK exists", lambda: assert_(hasattr(ft.ThemeMode, "DARK")))
check("ft.Button exists", lambda: ft.Button("test"))
check("ft.GestureDetector exists", lambda: assert_(hasattr(ft, "GestureDetector")))

# ── 5. Image sanity ──────────────────────────────────────────
print("\n[5/6] Image sanity")
import cv2
import numpy as np

def _check_placeholder():
    from src.ui.main_view import _placeholder_frame
    data = _placeholder_frame()
    assert isinstance(data, bytes) and len(data) > 100, "placeholder too small"

check("_placeholder_frame() produces valid JPEG bytes", _check_placeholder)

def _check_field_image():
    img = cv2.imread(str(config.FIELD_IMAGE_PATH))
    assert img is not None, "cannot read field image"
    h, w = img.shape[:2]
    ratio = w / h
    field_ratio = config.FIELD_LENGTH_M / config.FIELD_WIDTH_M
    assert abs(ratio - field_ratio) < 0.15, (
        f"aspect ratio mismatch: image={ratio:.2f}, field={field_ratio:.2f}"
    )

check("field_top_view.png aspect ratio ≈ field dimensions", _check_field_image)

def _check_robot_icons():
    for icon_path in [config.ROBOT_RED_ICON, config.ROBOT_BLUE_ICON]:
        img = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
        assert img is not None, f"cannot read {icon_path.name}"
        h, w = img.shape[:2]
        assert abs(w - h) < 5, f"{icon_path.name} not square: {w}x{h}"

check("robot icons are square", _check_robot_icons)

# ── 6. DuckDB schema ────────────────────────────────────────
print("\n[6/6] DuckDB schema")
from src.db.local_db import LocalDB

def _check_db():
    db = LocalDB()
    tables = [r[0] for r in db.query("SHOW TABLES")]
    for t in ["matches", "robot_tracks", "ball_tracks", "shot_events"]:
        assert t in tables, f"missing table: {t}"
    db.close()

check("DuckDB tables created correctly", _check_db)

# ── Summary ──────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91m{errors} check(s) failed.\033[0m")
    sys.exit(1)
else:
    print("\033[92mAll checks passed.\033[0m")


