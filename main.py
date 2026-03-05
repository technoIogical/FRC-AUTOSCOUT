"""FRC AutoScout – entry point.

Launch the Flet GUI application.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: F401  – initialises loguru
import flet as ft
from loguru import logger

from src.ui.app import build_app


def main() -> None:
    logger.info("Starting FRC AutoScout")
    ft.run(build_app)


if __name__ == "__main__":
    main()
