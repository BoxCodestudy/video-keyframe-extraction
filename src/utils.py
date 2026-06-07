"""Pure utility functions."""
import logging
import re
import time
from datetime import datetime
from pathlib import Path


def format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def format_event_time(seconds: float) -> str:
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    secs = (total_milliseconds % 60_000) // 1000
    millis = total_milliseconds % 1000
    return f"{hours:02d}{minutes:02d}{secs:02d}_{millis:03d}"


def estimate_eta(start_time: float, current: int, total: int) -> float:
    if current <= 0 or total <= 0 or current > total:
        return 0.0
    elapsed = max(time.time() - start_time, 0.001)
    rate = current / elapsed
    if rate <= 0:
        return 0.0
    return max((total - current) / rate, 0.0)


def extract_frame_index(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    if not match:
        return 0
    return int(match.group(1))


def sanitize_name(value: str) -> str:
    return re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE).strip("_") or "video"


def list_image_files(folder: Path) -> list[str]:
    return sorted([
        path.name for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])


def setup_logging(log_dir: Path, log_queue) -> tuple:
    """Configure root logger with GUI handler and file handler. Returns (gui_handler, file_handler)."""
    from src.config import LogHandler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    gui_handler = LogHandler(log_queue)
    gui_handler.setFormatter(formatter)
    logger.addHandler(gui_handler)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"video_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return gui_handler, file_handler
