"""Configuration management, constants, and data classes."""
import json
import logging
import queue
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ───────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = APP_DIR / "logs"
CONFIG_DIR = APP_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "video_processor_config.json"
TEMP_DIR = APP_DIR / "temp"
OUTPUT_DIR = APP_DIR / "output"

# ── Defaults ────────────────────────────────────────
DEFAULT_KEYFRAME_COUNT = 8
DEFAULT_SIMILARITY_THRESHOLD = 0.82
DEFAULT_ANALYSIS_WIDTH = 960
DEFAULT_ANALYSIS_HEIGHT = 540
DEFAULT_SAMPLING_STRIDE = 10
DEFAULT_INTRUSION_THRESHOLD = 0.45
DEFAULT_MIN_EVENT_SECONDS = 0.5
DEFAULT_MOTION_WEIGHT = 0.65
DEFAULT_PERSON_WEIGHT = 0.35
PROGRESS_POLL_MS = 120

# ── UI Theme ────────────────────────────────────────
ACCENT_COLOR = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
SUCCESS_COLOR = "#16a34a"
WARNING_COLOR = "#d97706"
SURFACE_COLOR = "#f5f7fa"
PANEL_COLOR = "#ffffff"
TEXT_COLOR = "#1e293b"
MUTED_COLOR = "#64748b"
BORDER_COLOR = "#e2e8f0"
CARD_RADIUS = 10
DANGER_COLOR = "#ef4444"


@dataclass
class ProcessingConfig:
    """Runtime processing parameters."""
    sample_stride: int = DEFAULT_SAMPLING_STRIDE
    intrusion_threshold: float = DEFAULT_INTRUSION_THRESHOLD
    min_event_duration: float = DEFAULT_MIN_EVENT_SECONDS
    motion_weight: float = DEFAULT_MOTION_WEIGHT
    person_weight: float = DEFAULT_PERSON_WEIGHT
    max_width: int = DEFAULT_ANALYSIS_WIDTH
    max_height: int = DEFAULT_ANALYSIS_HEIGHT
    keyframe_count: int = DEFAULT_KEYFRAME_COUNT
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD


@dataclass
class AppConfig:
    """Persistent application configuration."""
    output_path: str = ""
    temp_path: str = ""
    camera_save_path: str = ""
    sample_stride: int = DEFAULT_SAMPLING_STRIDE
    intrusion_threshold: float = DEFAULT_INTRUSION_THRESHOLD
    min_event_duration: float = DEFAULT_MIN_EVENT_SECONDS
    motion_weight: float = DEFAULT_MOTION_WEIGHT
    person_weight: float = DEFAULT_PERSON_WEIGHT
    max_width: int = DEFAULT_ANALYSIS_WIDTH
    auto_open_output: bool = True
    auto_export_video: bool = True
    remember_last_output: bool = True

    def __post_init__(self):
        if not self.output_path:
            self.output_path = str(OUTPUT_DIR)
        if not self.temp_path:
            self.temp_path = str(TEMP_DIR)
        if not self.camera_save_path:
            self.camera_save_path = str(APP_DIR / "saved_video")


@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class AnalysisResult:
    image_files: list[str]
    recommended_keyframes: list[str]
    feature_matrix: Optional["np.ndarray"] = None
    metadata: Optional[VideoMetadata] = None
    intrusion_events: list["IntrusionEvent"] = field(default_factory=list)
    sampling_stride: int = 10


@dataclass
class IntrusionEvent:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    max_score: float

    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 0.0)


class LogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def load_app_config() -> AppConfig:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return AppConfig(**{k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__})
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return AppConfig()


def save_app_config(config: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
