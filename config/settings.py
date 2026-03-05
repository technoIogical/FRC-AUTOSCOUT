from pathlib import Path
from loguru import logger
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model paths ──────────────────────────────────────────────
ROBOT_MODEL_PATH = BASE_DIR / "resources" / "models" / "robot.pt"
BALL_MODEL_PATH = BASE_DIR / "resources" / "models" / "ball.pt"

# ── Detection thresholds ─────────────────────────────────────
ROBOT_CONFIDENCE = 0.45
BALL_CONFIDENCE = 0.35

# ── Tracker ──────────────────────────────────────────────────
TRACK_THRESH = 0.25
TRACK_BUFFER = 30
MATCH_THRESH = 0.8

# ── Field dimensions (meters, 2024 FRC Crescendo) ───────────
FIELD_LENGTH_M = 16.54
FIELD_WIDTH_M = 8.21

# ── Robot real-world size (metres, including bumpers) ────────
ROBOT_SIZE_M = 0.85

# ── Field top-view image ─────────────────────────────────────
FIELD_IMAGE_PATH = BASE_DIR / "resources" / "images" / "field_top_view.png"
ROBOT_RED_ICON = BASE_DIR / "resources" / "images" / "robot_red.png"
ROBOT_BLUE_ICON = BASE_DIR / "resources" / "images" / "robot_blue.png"

# ── DuckDB ───────────────────────────────────────────────────
DUCKDB_PATH = BASE_DIR / "data" / "autoscout.duckdb"

# ── Supabase (set via environment variables) ─────────────────
SUPABASE_URL = ""
SUPABASE_KEY = ""

# ── Loguru configuration ─────────────────────────────────────
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
logger.add(LOG_DIR / "autoscout_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="7 days", level="DEBUG")
