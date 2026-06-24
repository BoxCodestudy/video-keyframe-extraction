# Code & UI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Split monolithic 1422-line `src/main.py` into 9 focused modules, add settings page, modernize UI, and apply performance fixes.

**Architecture:** Layered modules (config → algorithms → video I/O → GUI pages → app framework). Two Tkinter frames (MainPage, SettingsPage) switched within same window by VideoProcessorApp. Light theme, modern flat style, `#1a73e8` accent.

**Tech Stack:** Python 3, Tkinter/ttk, OpenCV, NumPy, Pillow, PyTorch, scikit-learn

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/config.py` | CREATE | `AppConfig`, `ProcessingConfig`, dataclasses, constants, config load/save |
| `src/utils.py` | CREATE | `sanitize_name`, `format_event_time`, `format_seconds`, `estimate_eta`, `extract_frame_index`, `list_image_files`, `enable_opencl_if_available`, logging setup |
| `src/detection.py` | CREATE | `detect_intrusion_events`, `evaluate_intrusion_metrics`, `map_events_to_image_files`, `resize_for_analysis` |
| `src/features.py` | CREATE | `CNNFeatureExtractor` with reusable ThreadPoolExecutor |
| `src/keyframes.py` | CREATE | `auto_select_keyframes_by_clustering`, `cosine_similarities`, `select_frames_by_anchor`, `evenly_spaced_keyframes` |
| `src/video_io.py` | CREATE | `video_to_images`, `build_segments_*`, `compose_video_*`, `generate_video_*`, `export_intrusion_clips`, `write_export_manifest`, `get_video_metadata` |
| `src/gui_settings.py` | CREATE | `SettingsPage(tk.Frame)` — output path, temp path, params, preferences |
| `src/gui_main.py` | CREATE | `MainPage(tk.Frame)` — redesigned 4-step pipeline UI |
| `src/gui_app.py` | CREATE | `VideoProcessorApp(tk.Tk)` — window, page switching, queue polling, config init |
| `src/main.py` | DELETE | Replaced by modules above |
| `requirements.txt` | CREATE | Pinned dependencies |
| `.gitignore` | CREATE | Python project ignores |

---

### Task 1: Create `src/config.py`

**Files:** Create `src/config.py`

Extract config, constants, and dataclasses from `src/main.py` lines 1-93.

- [ ] **Step 1: Write `src/config.py`**

```python
"""Configuration management, constants, and data classes."""
import json
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
ACCENT_COLOR = "#1a73e8"
ACCENT_HOVER = "#1557b0"
SUCCESS_COLOR = "#0ea854"
WARNING_COLOR = "#e37400"
SURFACE_COLOR = "#f0f2f5"
PANEL_COLOR = "#ffffff"
TEXT_COLOR = "#333333"
MUTED_COLOR = "#888888"
BORDER_COLOR = "#e0e4e8"
CARD_RADIUS = 10


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
    sample_stride: int = DEFAULT_SAMPLING_STRIDE
    intrusion_threshold: float = DEFAULT_INTRUSION_THRESHOLD
    min_event_duration: float = DEFAULT_MIN_EVENT_SECONDS
    motion_weight: float = DEFAULT_MOTION_WEIGHT
    person_weight: float = DEFAULT_PERSON_WEIGHT
    max_width: int = DEFAULT_ANALYSIS_WIDTH
    auto_open_output: bool = True
    remember_last_output: bool = True

    def __post_init__(self):
        if not self.output_path:
            self.output_path = str(OUTPUT_DIR)
        if not self.temp_path:
            self.temp_path = str(TEMP_DIR)


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
```

- [ ] **Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: extract config, constants, and dataclasses to src/config.py"
```

---

### Task 2: Create `src/utils.py`

**Files:** Create `src/utils.py`

Extract pure utility functions from `src/main.py` lines 125-167, 755-766.

- [ ] **Step 1: Write `src/utils.py`**

```python
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
    """Configure root logger with GUI handler and file handler. Returns handlers."""
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
```

- [ ] **Step 2: Add `LogHandler` to `src/config.py`** — append at end:

```python
import logging
import queue


class LogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))
```

- [ ] **Step 3: Commit**

```bash
git add src/utils.py src/config.py
git commit -m "feat: extract utility functions to src/utils.py"
```

---

### Task 3: Create `src/detection.py`

**Files:** Create `src/detection.py`

Extract intrusion detection logic from `src/main.py` lines 156-349 plus `evaluate_intrusion_metrics`, `map_events_to_image_files`.

- [ ] **Step 1: Write `src/detection.py`**

```python
"""Intrusion detection using MOG2 background subtraction + HOG pedestrian detection."""
import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import IntrusionEvent, DEFAULT_ANALYSIS_WIDTH, DEFAULT_ANALYSIS_HEIGHT
from src.utils import estimate_eta


def resize_for_analysis(frame: np.ndarray, max_width: int = DEFAULT_ANALYSIS_WIDTH, max_height: int = DEFAULT_ANALYSIS_HEIGHT) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def detect_intrusion_events(
    video_path: str,
    sampling_stride: int = 2,
    min_event_seconds: float = 0.8,
    cooldown_seconds: float = 0.6,
    motion_weight: float = 0.65,
    person_weight: float = 0.35,
    intrusion_threshold: float = 0.45,
    progress_callback: Optional[Callable[..., None]] = None,
) -> list[IntrusionEvent]:
    logger = logging.getLogger(__name__)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频进行入侵检测")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(sampling_stride))
    cooldown_frames = max(1, int(round(cooldown_seconds * fps)))
    min_duration = max(min_event_seconds, 0.05)
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=True)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    events: list[IntrusionEvent] = []
    current_event: Optional[dict] = None
    last_intrusion_frame = -cooldown_frames * 2
    frame_index = 0
    processed = 0
    start_time = time.time()
    prev_gray: Optional[np.ndarray] = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue
        processed += 1
        resized = resize_for_analysis(frame)
        fg_mask = mog.apply(resized)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(fg_mask)) / float(fg_mask.size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            diff_ratio = 0.0
        else:
            frame_diff = cv2.absdiff(gray, prev_gray)
            _, diff_mask = cv2.threshold(frame_diff, 18, 255, cv2.THRESH_BINARY)
            diff_ratio = float(np.count_nonzero(diff_mask)) / float(diff_mask.size)
        prev_gray = gray
        # Skip HOG on tiny frames to avoid errors
        if gray.shape[0] < 128 or gray.shape[1] < 128:
            person_signal = 0.0
        else:
            try:
                person_rects, person_weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                person_confidence = float(max(person_weights) if len(person_weights) else 0.0)
                person_signal = 1.0 if person_confidence >= 0.2 or len(person_rects) > 0 else 0.0
            except cv2.error:
                person_signal = 0.0
        # Skip HOG for very low motion frames (performance optimization)
        motion_signal_raw = float(np.clip((motion_ratio - 0.015) / 0.12, 0.0, 1.0))
        diff_signal_raw = float(np.clip((diff_ratio - 0.01) / 0.08, 0.0, 1.0))
        if motion_signal_raw < 0.03 and diff_signal_raw < 0.03:
            person_signal = 0.0  # Skip HOG result for near-static frames
        motion_signal = max(motion_signal_raw, diff_signal_raw)
        intrusion_score = motion_weight * motion_signal + person_weight * person_signal
        is_intrusion = intrusion_score >= intrusion_threshold and (motion_ratio >= 0.008 or diff_ratio >= 0.008)
        if is_intrusion:
            if current_event is None:
                current_event = {"start": frame_index, "end": frame_index, "max_score": intrusion_score}
            else:
                current_event["end"] = frame_index
                current_event["max_score"] = max(current_event["max_score"], intrusion_score)
            last_intrusion_frame = frame_index
        if current_event is not None and frame_index - last_intrusion_frame >= cooldown_frames:
            start_frame = int(current_event["start"])
            end_frame = int(current_event["end"])
            event = IntrusionEvent(
                start_frame=start_frame, end_frame=end_frame,
                start_time=start_frame / fps, end_time=end_frame / fps,
                max_score=float(current_event["max_score"]),
            )
            if event.duration >= min_duration:
                events.append(event)
            current_event = None
        if progress_callback and processed % 4 == 0:
            progress_callback(
                stage="detecting_intrusion",
                current=min(frame_index + 1, total_frames) if total_frames else frame_index + 1,
                total=total_frames or frame_index + 1,
                eta_seconds=estimate_eta(start_time, min(frame_index + 1, total_frames) if total_frames else frame_index + 1, total_frames or frame_index + 1),
                found_count=len(events) + (1 if current_event else 0),
            )
        frame_index += 1
    cap.release()
    if current_event is not None:
        start_frame = int(current_event["start"])
        end_frame = int(current_event["end"])
        event = IntrusionEvent(
            start_frame=start_frame, end_frame=end_frame,
            start_time=start_frame / fps, end_time=end_frame / fps,
            max_score=float(current_event["max_score"]),
        )
        if event.duration >= min_duration:
            events.append(event)
    events.sort(key=lambda item: item.start_frame)
    logger.info("入侵检测完成，共识别到 %s 个事件片段", len(events))
    if progress_callback:
        progress_callback(
            stage="detecting_intrusion", current=total_frames or frame_index,
            total=total_frames or frame_index, eta_seconds=0.0, found_count=len(events),
        )
    return events


def evaluate_intrusion_metrics(predicted_events: list[IntrusionEvent], ground_truth_events: list[IntrusionEvent]) -> tuple[float, float]:
    if not ground_truth_events:
        if not predicted_events:
            return 1.0, 1.0
        return 0.0, 1.0
    true_positive = 0
    matched_gt: set[int] = set()
    for pred in predicted_events:
        for gt_index, gt in enumerate(ground_truth_events):
            if gt_index in matched_gt:
                continue
            overlap_start = max(pred.start_time, gt.start_time)
            overlap_end = min(pred.end_time, gt.end_time)
            overlap = max(overlap_end - overlap_start, 0.0)
            gt_duration = max(gt.duration, 1e-6)
            if overlap / gt_duration >= 0.3:
                true_positive += 1
                matched_gt.add(gt_index)
                break
    precision = true_positive / max(len(predicted_events), 1)
    recall = true_positive / max(len(ground_truth_events), 1)
    return precision, recall


def map_events_to_image_files(image_files: list[str], events: list[IntrusionEvent]) -> list[str]:
    from src.utils import extract_frame_index
    if not events:
        return []
    selected = []
    for image_name in image_files:
        frame_index = extract_frame_index(image_name)
        for event in events:
            if event.start_frame <= frame_index <= event.end_frame:
                selected.append(image_name)
                break
    return selected
```

- [ ] **Step 2: Commit**

```bash
git add src/detection.py
git commit -m "feat: extract intrusion detection to src/detection.py"
```

---

### Task 4: Create `src/features.py`

**Files:** Create `src/features.py`

Extract `CNNFeatureExtractor` from `src/main.py` lines 352-398. Fix: reuse thread pool across batches instead of creating new one each iteration.

- [ ] **Step 1: Write `src/features.py`**

```python
"""CNN feature extraction using ResNet-18."""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
from PIL import Image

from src.utils import estimate_eta

try:
    import torch
    from torchvision import models
    from torchvision.models import ResNet18_Weights
    HAS_ML = True
except ImportError:
    torch = None
    models = None
    ResNet18_Weights = None
    HAS_ML = False


class CNNFeatureExtractor:
    def __init__(self, batch_size: int = 16):
        if not HAS_ML:
            raise RuntimeError("当前环境未安装机器学习依赖")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        self.transform = weights.transforms()
        self.batch_size = batch_size if self.device.type == "cuda" else min(batch_size, 8)
        # Reusable thread pool (fix: was created per batch)
        self._pool = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)))

    def _load_tensor(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def extract_batch(
        self,
        image_paths: list[str],
        progress_callback: Optional[Callable[..., None]] = None,
        found_count: int = 0,
    ) -> np.ndarray:
        if not image_paths:
            return np.empty((0, 512), dtype=np.float32)
        features = []
        total = len(image_paths)
        start_time = time.time()
        for start in range(0, total, self.batch_size):
            batch_paths = image_paths[start:start + self.batch_size]
            tensors = list(self._pool.map(self._load_tensor, batch_paths))
            batch_tensor = torch.stack(tensors).to(self.device, non_blocking=self.device.type == "cuda")
            with torch.no_grad():
                batch_features = self.model(batch_tensor).detach().cpu().numpy()
            features.append(batch_features)
            current = min(start + len(batch_paths), total)
            if progress_callback:
                progress_callback(
                    stage="recognizing", current=current, total=total,
                    eta_seconds=estimate_eta(start_time, current, total),
                    found_count=found_count,
                )
        return np.concatenate(features, axis=0)

    def close(self):
        self._pool.shutdown(wait=False)
```

- [ ] **Step 2: Commit**

```bash
git add src/features.py
git commit -m "feat: extract CNN feature extractor to src/features.py with pool reuse"
```

---

### Task 5: Create `src/keyframes.py`

**Files:** Create `src/keyframes.py`

Extract keyframe selection from `src/main.py` lines 165-458, 519-555.

- [ ] **Step 1: Write `src/keyframes.py`**

```python
"""Keyframe selection via KMeans clustering and anchor-based similarity filtering."""
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import AnalysisResult, IntrusionEvent, VideoMetadata, DEFAULT_KEYFRAME_COUNT, DEFAULT_SIMILARITY_THRESHOLD
from src.features import HAS_ML, CNNFeatureExtractor, KMeans
from src.utils import extract_frame_index, list_image_files

# Conditional sklearn import
try:
    from sklearn.cluster import KMeans as KMeansImpl
except ImportError:
    KMeansImpl = None


def evenly_spaced_keyframes(image_files: list[str], keyframe_count: int) -> list[str]:
    if not image_files:
        return []
    if len(image_files) <= keyframe_count:
        return image_files
    positions = np.linspace(0, len(image_files) - 1, keyframe_count, dtype=int)
    selected = [image_files[index] for index in positions]
    return sorted(dict.fromkeys(selected), key=extract_frame_index)


def auto_select_keyframes_by_clustering(
    image_folder: str,
    n_clusters: int = DEFAULT_KEYFRAME_COUNT,
    progress_callback: Optional[Callable[..., None]] = None,
    feature_extractor: Optional[CNNFeatureExtractor] = None,
    metadata: Optional[VideoMetadata] = None,
    allowed_images: Optional[list[str]] = None,
    intrusion_events: Optional[list[IntrusionEvent]] = None,
    sampling_stride: int = 1,
) -> AnalysisResult:
    logger = logging.getLogger(__name__)
    folder = Path(image_folder)
    image_files = list_image_files(folder)
    allowed_set = set(allowed_images or [])
    candidate_images = image_files if not allowed_set else [name for name in image_files if name in allowed_set]
    if not candidate_images:
        candidate_images = image_files
    if metadata is None:
        metadata = VideoMetadata(path="", fps=24.0, frame_count=len(image_files), width=0, height=0)
    if not image_files:
        return AnalysisResult(image_files=[], recommended_keyframes=[], metadata=metadata,
                              intrusion_events=intrusion_events or [], sampling_stride=sampling_stride)
    target_clusters = max(1, min(n_clusters, len(candidate_images)))
    if not HAS_ML or KMeansImpl is None:
        keyframes = evenly_spaced_keyframes(candidate_images, target_clusters)
        if progress_callback:
            progress_callback(stage="recognizing", current=len(candidate_images), total=len(candidate_images),
                              eta_seconds=0, found_count=len(keyframes))
        logger.warning("机器学习依赖不可用，已回退到均匀抽样关键帧策略")
        return AnalysisResult(image_files=image_files, recommended_keyframes=keyframes, metadata=metadata,
                              intrusion_events=intrusion_events or [], sampling_stride=sampling_stride)
    extractor = feature_extractor or CNNFeatureExtractor()
    feature_paths = [str(folder / image_name) for image_name in candidate_images]
    features = extractor.extract_batch(feature_paths, progress_callback=progress_callback)
    if len(features) == 0:
        keyframes = evenly_spaced_keyframes(candidate_images, target_clusters)
        return AnalysisResult(image_files=image_files, recommended_keyframes=keyframes, metadata=metadata,
                              intrusion_events=intrusion_events or [], sampling_stride=sampling_stride)
    kmeans = KMeansImpl(n_clusters=target_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    keyframes = []
    for cluster_id in range(target_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_vectors = features[cluster_indices]
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        best_index = cluster_indices[int(np.argmin(distances))]
        keyframes.append(candidate_images[int(best_index)])
    keyframes = sorted(dict.fromkeys(keyframes), key=extract_frame_index)
    if progress_callback:
        progress_callback(stage="recognizing", current=len(candidate_images), total=len(candidate_images),
                          eta_seconds=0, found_count=len(keyframes))
    logger.info("关键帧识别完成，共推荐 %s 张锚点图", len(keyframes))
    return AnalysisResult(image_files=candidate_images, recommended_keyframes=keyframes, feature_matrix=features,
                          metadata=metadata, intrusion_events=intrusion_events or [], sampling_stride=sampling_stride)


def cosine_similarities(feature_matrix: np.ndarray, anchor_index: int) -> np.ndarray:
    anchor_vector = feature_matrix[anchor_index]
    anchor_norm = np.linalg.norm(anchor_vector) + 1e-8
    matrix_norm = np.linalg.norm(feature_matrix, axis=1) + 1e-8
    return (feature_matrix @ anchor_vector) / (matrix_norm * anchor_norm)


def select_frames_by_anchor(
    anchor_name: str, image_folder: str, analysis_result: Optional[AnalysisResult],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[str]:
    folder = Path(image_folder)
    if analysis_result and analysis_result.feature_matrix is not None and anchor_name in analysis_result.image_files:
        anchor_index = analysis_result.image_files.index(anchor_name)
        similarities = cosine_similarities(analysis_result.feature_matrix, anchor_index)
        selected = [image_name for image_name, score in zip(analysis_result.image_files, similarities) if float(score) >= threshold]
        return sorted(selected or [anchor_name], key=extract_frame_index)
    selected = []
    anchor = cv2.imread(str(folder / anchor_name))
    if anchor is None:
        return [anchor_name]
    anchor_hist = cv2.calcHist([anchor], [0], None, [256], [0, 256])
    for image_name in list_image_files(folder):
        frame = cv2.imread(str(folder / image_name))
        if frame is None:
            continue
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        score = cv2.compareHist(anchor_hist, hist, cv2.HISTCMP_CORREL)
        if score >= 0.8:
            selected.append(image_name)
    return sorted(selected or [anchor_name], key=extract_frame_index)
```

- [ ] **Step 2: Commit**

```bash
git add src/keyframes.py
git commit -m "feat: extract keyframe selection to src/keyframes.py"
```

---

### Task 6: Create `src/video_io.py`

**Files:** Create `src/video_io.py`

Extract video I/O from `src/main.py` lines 175-816.

- [ ] **Step 1: Write `src/video_io.py`**

```python
"""Video I/O: decomposition, composition via FFmpeg/OpenCV, export utilities."""
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

from src.config import VideoMetadata, IntrusionEvent, DEFAULT_ANALYSIS_WIDTH, DEFAULT_ANALYSIS_HEIGHT
from src.detection import resize_for_analysis
from src.utils import estimate_eta, sanitize_name, format_event_time

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def enable_opencl_if_available() -> None:
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    except cv2.error:
        pass


def get_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("视频无法打开，请检查编解码器或文件格式")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoMetadata(path=video_path, fps=fps, frame_count=frame_count, width=width, height=height)


def video_to_images(
    video_path: str, output_folder: str, sampling_rate: int,
    progress_callback: Optional[Callable[..., None]] = None,
    max_width: int = DEFAULT_ANALYSIS_WIDTH, max_height: int = DEFAULT_ANALYSIS_HEIGHT,
) -> VideoMetadata:
    logger = logging.getLogger(__name__)
    output_dir = Path(output_folder)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enable_opencl_if_available()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("视频无法打开，请检查编解码器或文件格式")
    metadata = VideoMetadata(
        path=video_path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 24.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    stride = max(1, int(sampling_rate))
    frame_id = 0
    saved_count = 0
    start_time = time.time()
    logger.info("开始视频分解，采样率=%s，总帧数=%s", stride, metadata.frame_count)
    # Performance: use CAP_PROP_POS_FRAMES to skip frames instead of decoding all
    while frame_id < metadata.frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            break
        image_name = f"frame_{frame_id:06d}.jpg"
        image_path = output_dir / image_name
        preview_frame = resize_for_analysis(frame, max_width=max_width, max_height=max_height)
        cv2.imwrite(str(image_path), preview_frame)
        saved_count += 1
        current = min(frame_id + 1, metadata.frame_count)
        if progress_callback and saved_count % 60 == 0:
            progress_callback(
                stage="extracting", current=current, total=metadata.frame_count,
                eta_seconds=estimate_eta(start_time, current, metadata.frame_count),
                saved_count=saved_count,
            )
        frame_id += stride
    cap.release()
    if progress_callback:
        progress_callback(stage="extracting", current=metadata.frame_count or frame_id,
                          total=metadata.frame_count or frame_id, eta_seconds=0, saved_count=saved_count)
    logger.info("视频分解完成，共保存 %s 张分析帧", saved_count)
    return metadata


def build_segments_from_frames(selected_frames: list[str], fps: float, frame_count: int,
                               padding_seconds: float = 0.5, merge_seconds: float = 1.0) -> list[tuple[int, int]]:
    from src.utils import extract_frame_index
    if not selected_frames:
        return []
    padding = max(1, int(round(fps * padding_seconds)))
    merge_gap = max(1, int(round(fps * merge_seconds)))
    frame_indices = sorted({extract_frame_index(name) for name in selected_frames})
    segments: list[tuple[int, int]] = []
    for index in frame_indices:
        start = max(0, index - padding)
        end = min(max(frame_count - 1, 0), index + padding)
        if segments and start <= segments[-1][1] + merge_gap:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end))
        else:
            segments.append((start, end))
    return segments


def build_segments_from_events(events: list[IntrusionEvent], frame_count: int) -> list[tuple[int, int]]:
    if not events:
        return []
    segments = []
    for event in sorted(events, key=lambda item: item.start_frame):
        start = max(0, int(event.start_frame))
        end = min(max(frame_count - 1, 0), int(event.end_frame))
        if segments and start <= segments[-1][1] + 1:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end))
        else:
            segments.append((start, end))
    return segments


def _get_ffmpeg_exe() -> Optional[str]:
    if imageio_ffmpeg is None:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _compose_with_ffmpeg(source_video_path: str, output_path: str, segments: list[tuple[int, int]], fps: float) -> bool:
    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe or not segments:
        return False
    filters_with_audio = []
    filters_video_only = []
    concat_inputs_audio = []
    concat_inputs_video = []
    for index, (frame_start, frame_end) in enumerate(segments):
        start_time = max(frame_start / max(fps, 0.001), 0)
        end_time = max((frame_end + 1) / max(fps, 0.001), start_time + 0.001)
        filters_with_audio.append(f"[0:v]trim=start={start_time}:end={end_time},setpts=PTS-STARTPTS[v{index}]")
        filters_with_audio.append(f"[0:a]atrim=start={start_time}:end={end_time},asetpts=PTS-STARTPTS[a{index}]")
        filters_video_only.append(f"[0:v]trim=start={start_time}:end={end_time},setpts=PTS-STARTPTS[v{index}]")
        concat_inputs_audio.append(f"[v{index}][a{index}]")
        concat_inputs_video.append(f"[v{index}]")
    # Try with audio first
    audio_filter = ";".join(filters_with_audio + [f"{''.join(concat_inputs_audio)}concat=n={len(segments)}:v=1:a=1[v][a]"])
    result = subprocess.run([ffmpeg_exe, "-y", "-i", source_video_path, "-filter_complex", audio_filter,
                             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", output_path],
                            capture_output=True, text=True)
    if result.returncode == 0 and Path(output_path).exists():
        return True
    # Fallback: video only
    video_filter = ";".join(filters_video_only + [f"{''.join(concat_inputs_video)}concat=n={len(segments)}:v=1:a=0[v]"])
    result = subprocess.run([ffmpeg_exe, "-y", "-i", source_video_path, "-filter_complex", video_filter,
                             "-map", "[v]", "-c:v", "libx264", "-preset", "fast", output_path],
                            capture_output=True, text=True)
    return result.returncode == 0 and Path(output_path).exists()


def _compose_with_opencv(source_video_path: str, output_path: str, segments: list[tuple[int, int]], metadata: VideoMetadata) -> None:
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise RuntimeError("无法重新读取原始视频用于生成结果")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, metadata.fps, (metadata.width, metadata.height))
    current_segment = 0
    frame_id = 0
    while cap.isOpened() and current_segment < len(segments):
        ok, frame = cap.read()
        if not ok:
            break
        segment_start, segment_end = segments[current_segment]
        if frame_id < segment_start:
            frame_id += 1
            continue
        if frame_id > segment_end:
            current_segment += 1
            continue
        writer.write(frame)
        frame_id += 1
    writer.release()
    cap.release()


def generate_video_from_segments(source_video_path: str, segments: list[tuple[int, int]],
                                 output_path: str, metadata: VideoMetadata) -> list[tuple[int, int]]:
    if not segments:
        raise ValueError("没有可用的关键帧片段生成视频")
    success = _compose_with_ffmpeg(source_video_path, output_path, segments, metadata.fps)
    if not success:
        _compose_with_opencv(source_video_path, output_path, segments, metadata)
    return segments


def generate_video_from_selection(source_video_path: str, selected_frames: list[str],
                                  output_path: str, metadata: VideoMetadata) -> list[tuple[int, int]]:
    segments = build_segments_from_frames(selected_frames, fps=metadata.fps, frame_count=metadata.frame_count)
    return generate_video_from_segments(source_video_path, segments, output_path, metadata)


def event_to_segment(event: IntrusionEvent) -> tuple[int, int]:
    return event.start_frame, event.end_frame


def export_intrusion_clips(source_video_path: str, metadata: VideoMetadata, events: list[IntrusionEvent],
                           events_root_directory: Path, file_stem: str, export_stamp: str) -> list[str]:
    events_root_directory.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for event_index, event in enumerate(events, start=1):
        start_label = format_event_time(event.start_time)
        end_label = format_event_time(event.end_time)
        event_folder_name = f"event_{event_index:03d}__intrusion__{start_label}_to_{end_label}__{export_stamp}"
        event_folder = events_root_directory / event_folder_name
        event_folder.mkdir(parents=True, exist_ok=True)
        clip_name = f"{file_stem}__event_{event_index:03d}__intrusion__{start_label}_to_{end_label}__{export_stamp}.mp4"
        clip_path = event_folder / clip_name
        generate_video_from_segments(source_video_path, [event_to_segment(event)], str(clip_path), metadata)
        generated_files.append(str(clip_path))
    return generated_files


def build_export_paths(base_save_path: str, source_video_path: str) -> dict:
    save_target = Path(base_save_path)
    export_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = sanitize_name(Path(source_video_path).stem)
    root_name = f"{video_stem}__intrusion_export__{export_stamp}"
    root_dir = save_target.parent / root_name
    master_dir = root_dir / "00_master"
    events_dir = root_dir / "01_events"
    index_dir = root_dir / "02_indexes"
    master_name = f"{video_stem}__intrusion_summary__{export_stamp}.mp4"
    return {
        "root_dir": root_dir, "master_dir": master_dir, "events_dir": events_dir,
        "index_dir": index_dir, "master_file": master_dir / master_name,
        "video_stem": video_stem, "export_stamp": export_stamp,
    }


def write_export_manifest(export_root: Path, master_path: Path, event_files: list[str],
                          events: list[IntrusionEvent], source_video_path: str) -> Path:
    manifest_data = {
        "source_video": source_video_path,
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "master_video": str(master_path),
        "event_count": len(events),
        "event_files": event_files,
        "events": [{"index": i + 1, "start_time": e.start_time, "end_time": e.end_time,
                     "duration": e.duration, "max_score": e.max_score} for i, e in enumerate(events)],
    }
    manifest_path = export_root / "02_indexes" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
```

- [ ] **Step 2: Commit**

```bash
git add src/video_io.py
git commit -m "feat: extract video I/O to src/video_io.py with frame-skip optimization"
```

---

### Task 7: Create `src/gui_settings.py`

**Files:** Create `src/gui_settings.py`

Settings page as a Tkinter Frame. Paths, processing params, UI preferences.

- [ ] **Step 1: Write `src/gui_settings.py`**

```python
"""Settings page — output paths, processing parameters, UI preferences."""
import tkinter as tk
from tkinter import ttk, filedialog
from src.config import AppConfig, save_app_config, ACCENT_COLOR, SURFACE_COLOR, PANEL_COLOR, TEXT_COLOR, MUTED_COLOR, BORDER_COLOR


class SettingsPage(tk.Frame):
    def __init__(self, parent, app_config: AppConfig, on_back: callable, on_save: callable):
        super().__init__(parent, bg=SURFACE_COLOR)
        self.app_config = app_config
        self.on_back = on_back
        self.on_save = on_save
        self._build_ui()
        self._load_config()

    def _build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self, bg=PANEL_COLOR, height=48)
        topbar.pack(fill="x", padx=0, pady=0)
        topbar.pack_propagate(False)
        back_btn = tk.Button(topbar, text="← 返回", bg=PANEL_COLOR, fg=TEXT_COLOR,
                             font=("Microsoft YaHei", 10), bd=0, cursor="hand2",
                             activebackground=PANEL_COLOR, command=self.on_back)
        back_btn.pack(side="left", padx=16, pady=10)
        title = tk.Label(topbar, text="⚙ 系统设置", bg=PANEL_COLOR, fg=TEXT_COLOR,
                         font=("Microsoft YaHei", 14, "bold"))
        title.pack(side="left", padx=(0, 16))
        save_btn = tk.Button(topbar, text="保存设置", bg=ACCENT_COLOR, fg="white",
                             font=("Microsoft YaHei", 10, "bold"), bd=0, padx=16, pady=6,
                             cursor="hand2", activebackground="#1557b0", command=self._save)
        save_btn.pack(side="right", padx=16, pady=10)

        # ── Scrollable content ──
        canvas = tk.Canvas(self, bg=SURFACE_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.content = tk.Frame(canvas, bg=SURFACE_COLOR)
        self.content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.content, anchor="nw", tags="self.content")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # Mouse wheel scrolling (Windows)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # ── Section: Paths ──
        self._add_section("文件路径")
        self.output_var = tk.StringVar()
        self.temp_var = tk.StringVar()
        self._add_path_row("输出文件默认路径", self.output_var, "所有处理结果将保存到此目录")
        self._add_path_row("临时文件路径", self.temp_var, "视频分解时的帧缓存目录，处理完成后可自动清理")

        # ── Section: Parameters ──
        self._add_section("默认处理参数")
        self.stride_var = tk.StringVar()
        self.threshold_var = tk.StringVar()
        self.maxw_var = tk.StringVar()
        self.minevent_var = tk.StringVar()
        self.motion_var = tk.StringVar()
        self.person_var = tk.StringVar()
        grid = tk.Frame(self.content, bg=SURFACE_COLOR)
        grid.pack(fill="x", padx=24, pady=(4, 0))
        self._add_param_row(grid, 0, "采样帧间隔", self.stride_var)
        self._add_param_row(grid, 0, "入侵检测阈值", self.threshold_var)
        self._add_param_row(grid, 1, "最大分辨率宽度", self.maxw_var)
        self._add_param_row(grid, 1, "最小事件时长(秒)", self.minevent_var)
        self._add_param_row(grid, 2, "运动检测权重", self.motion_var)
        self._add_param_row(grid, 2, "人物检测权重", self.person_var)

        # ── Section: Preferences ──
        self._add_section("界面偏好")
        pref_frame = tk.Frame(self.content, bg=SURFACE_COLOR)
        pref_frame.pack(fill="x", padx=24, pady=(4, 20))
        self.auto_open_var = tk.BooleanVar()
        self.remember_var = tk.BooleanVar()
        tk.Checkbutton(pref_frame, text="处理完成后自动打开输出文件夹", variable=self.auto_open_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)
        tk.Checkbutton(pref_frame, text="启动时自动加载上次的输出路径", variable=self.remember_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)

    def _add_section(self, title: str):
        sep = tk.Frame(self.content, bg=BORDER_COLOR, height=1)
        sep.pack(fill="x", padx=24, pady=(16, 8))
        tk.Label(self.content, text=title, bg=SURFACE_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=24, pady=(0, 4))

    def _add_path_row(self, label: str, var: tk.StringVar, hint: str):
        row = tk.Frame(self.content, bg=SURFACE_COLOR)
        row.pack(fill="x", padx=24, pady=4)
        tk.Label(row, text=label, bg=SURFACE_COLOR, fg=TEXT_COLOR,
                 font=("Microsoft YaHei", 10)).pack(anchor="w")
        entry_row = tk.Frame(row, bg=SURFACE_COLOR)
        entry_row.pack(fill="x", pady=(2, 0))
        entry = tk.Entry(entry_row, textvariable=var, font=("Microsoft YaHei", 10),
                         bg=PANEL_COLOR, fg=TEXT_COLOR, relief="solid", bd=1)
        entry.pack(side="left", fill="x", expand=True, ipady=4)
        browse_btn = tk.Button(entry_row, text="浏览...", bg=PANEL_COLOR, fg=ACCENT_COLOR,
                               font=("Microsoft YaHei", 9), bd=1, padx=10, cursor="hand2",
                               command=lambda v=var: self._browse_dir(v))
        browse_btn.pack(side="left", padx=(6, 0))
        tk.Label(row, text=hint, bg=SURFACE_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 8)).pack(anchor="w", pady=(1, 0))

    def _add_param_row(self, parent, col, label, var):
        f = tk.Frame(parent, bg=SURFACE_COLOR)
        f.grid(row=col, column=0, sticky="w", padx=(0, 30), pady=4)
        tk.Label(f, text=label, bg=SURFACE_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 9)).pack(side="left")
        tk.Entry(f, textvariable=var, width=8, font=("Microsoft YaHei", 10),
                 bg=PANEL_COLOR, fg=TEXT_COLOR, relief="solid", bd=1).pack(side="left", padx=(4, 0), ipady=2)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(title="选择目录")
        if path:
            var.set(path)

    def _load_config(self):
        c = self.app_config
        self.output_var.set(c.output_path)
        self.temp_var.set(c.temp_path)
        self.stride_var.set(str(c.sample_stride))
        self.threshold_var.set(str(c.intrusion_threshold))
        self.maxw_var.set(str(c.max_width))
        self.minevent_var.set(str(c.min_event_duration))
        self.motion_var.set(str(c.motion_weight))
        self.person_var.set(str(c.person_weight))
        self.auto_open_var.set(c.auto_open_output)
        self.remember_var.set(c.remember_last_output)

    def _save(self):
        try:
            self.app_config.output_path = self.output_var.get()
            self.app_config.temp_path = self.temp_var.get()
            self.app_config.sample_stride = int(self.stride_var.get())
            self.app_config.intrusion_threshold = float(self.threshold_var.get())
            self.app_config.max_width = int(self.maxw_var.get())
            self.app_config.min_event_duration = float(self.minevent_var.get())
            self.app_config.motion_weight = float(self.motion_var.get())
            self.app_config.person_weight = float(self.person_var.get())
            self.app_config.auto_open_output = self.auto_open_var.get()
            self.app_config.remember_last_output = self.remember_var.get()
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("错误", "参数格式不正确，请检查数值输入")
            return
        save_app_config(self.app_config)
        self.on_save()
```

- [ ] **Step 2: Commit**

```bash
git add src/gui_settings.py
git commit -m "feat: add settings page with path config and parameter preferences"
```

---

### Task 8: Create `src/gui_main.py`

**Files:** Create `src/gui_main.py`

Redesigned main processing page. Left step panel with status, center preview + stats, bottom log.

- [ ] **Step 1: Write `src/gui_main.py`**

Redesigned main page with:
- Top bar: logo, "⚙ 设置" / "📂 选择视频" / "▶ 开始处理"
- Left: 4 step cards with status (waiting/active/done), each with inline progress
- Right-top: preview area with drag-drop hint
- Right-mid: stats row (events, keyframes, time, output path)
- Right-bottom: collapsible log panel
- Bottom status bar

```python
"""Main processing page — video pipeline UI."""
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from PIL import Image, ImageTk

from src.config import (AppConfig, ProcessingConfig, AnalysisResult, VideoMetadata,
                        IntrusionEvent, SURFACE_COLOR, PANEL_COLOR, TEXT_COLOR,
                        MUTED_COLOR, ACCENT_COLOR, SUCCESS_COLOR, WARNING_COLOR,
                        BORDER_COLOR, CARD_RADIUS, PROGRESS_POLL_MS)
from src.detection import detect_intrusion_events, map_events_to_image_files
from src.features import CNNFeatureExtractor
from src.keyframes import auto_select_keyframes_by_clustering, select_frames_by_anchor
from src.video_io import (get_video_metadata, video_to_images, generate_video_from_segments,
                          build_segments_from_events, build_export_paths,
                          export_intrusion_clips, write_export_manifest)
from src.utils import format_seconds, list_image_files


class MainPage(tk.Frame):
    def __init__(self, parent, app_config: AppConfig, processing_config: ProcessingConfig,
                 log_queue: queue.Queue, ui_queue: queue.Queue,
                 on_open_settings: callable):
        super().__init__(parent, bg=SURFACE_COLOR)
        self.app_config = app_config
        self.proc_config = processing_config
        self.log_queue = log_queue
        self.ui_queue = ui_queue
        self.on_open_settings = on_open_settings

        self.video_path = tk.StringVar()
        self.status_text = tk.StringVar(value="就绪")
        self.status_extra = tk.StringVar(value="无视频加载")
        self.preview_photo = None
        self.pipeline_thread: Optional[threading.Thread] = None
        self.analysis_result: Optional[AnalysisResult] = None
        self.current_metadata: Optional[VideoMetadata] = None
        self.current_selected_frames: list[str] = []
        self.current_intrusion_events: list[IntrusionEvent] = []
        self.source_sampled_frames: list[str] = []

        # Step tracking
        self.step_state = {"1": "waiting", "2": "waiting", "3": "waiting", "4": "waiting"}
        self.step_desc = {"1": tk.StringVar(value="等待开始"), "2": tk.StringVar(value="等待开始"),
                          "3": tk.StringVar(value="等待开始"), "4": tk.StringVar(value="等待开始")}
        self.step_progress = {"1": tk.DoubleVar(value=0), "2": tk.DoubleVar(value=0),
                              "3": tk.DoubleVar(value=0), "4": tk.DoubleVar(value=0)}

        # Stats
        self.stat_events = tk.StringVar(value="--")
        self.stat_keyframes = tk.StringVar(value="--")
        self.stat_time = tk.StringVar(value="--")
        self.stat_output = tk.StringVar(value="--")

        # Log collapsed state
        self.log_expanded = tk.BooleanVar(value=True)
        # Preview frame index
        self._preview_index = 0

        self._build_ui()
        self._poll_log()

    def _build_ui(self):
        # ── Top Bar ──
        topbar = tk.Frame(self, bg=PANEL_COLOR)
        topbar.pack(fill="x", padx=0, pady=0)
        logo = tk.Label(topbar, text="◈ 视频关键帧筛选与重构系统", bg=PANEL_COLOR, fg=ACCENT_COLOR,
                        font=("Microsoft YaHei", 13, "bold"))
        logo.pack(side="left", padx=16, pady=10)
        btn_frame = tk.Frame(topbar, bg=PANEL_COLOR)
        btn_frame.pack(side="right", padx=12, pady=8)
        self.settings_btn = tk.Button(btn_frame, text="⚙ 设置", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                      font=("Microsoft YaHei", 10), bd=1, relief="solid",
                                      padx=10, pady=4, cursor="hand2", command=self.on_open_settings)
        self.settings_btn.pack(side="left", padx=4)
        self.select_btn = tk.Button(btn_frame, text="📂 选择视频", bg=PANEL_COLOR, fg=WARNING_COLOR,
                                    font=("Microsoft YaHei", 10, "bold"), bd=1, relief="solid",
                                    padx=10, pady=4, cursor="hand2", command=self._select_video)
        self.select_btn.pack(side="left", padx=4)
        self.start_btn = tk.Button(btn_frame, text="▶ 开始处理", bg=ACCENT_COLOR, fg="white",
                                   font=("Microsoft YaHei", 10, "bold"), bd=0,
                                   padx=14, pady=4, cursor="hand2", command=self._start_processing)
        self.start_btn.pack(side="left", padx=4)
        self.start_btn.config(state="disabled")

        # ── Main Content ──
        main = tk.Frame(self, bg=SURFACE_COLOR)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Left: Steps ──
        left_panel = tk.Frame(main, bg=PANEL_COLOR, width=260)
        left_panel.pack(side="left", fill="y", padx=(0, 8))
        left_panel.pack_propagate(False)
        tk.Label(left_panel, text="处理流程", bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 8))
        step_names = ["视频分解与采样", "入侵检测", "关键帧筛选", "视频重构"]
        self.step_frames = {}
        for i, name in enumerate(step_names, 1):
            self.step_frames[str(i)] = self._create_step_card(left_panel, str(i), name)

        # ── Right: Preview + Stats + Log ──
        right_panel = tk.Frame(main, bg=SURFACE_COLOR)
        right_panel.pack(side="left", fill="both", expand=True)

        # Preview area
        preview_frame = tk.Frame(right_panel, bg=PANEL_COLOR, bd=0, highlightthickness=0)
        preview_frame.pack(fill="both", expand=True, pady=(0, 4))
        self.preview_label = tk.Label(preview_frame, bg=PANEL_COLOR, anchor="center",
                                       text="🎬\n拖拽视频文件到此处 或 点击"选择视频"开始\n支持 MP4 / AVI / MKV / MOV",
                                       fg=MUTED_COLOR, font=("Microsoft YaHei", 12))
        self.preview_label.pack(fill="both", expand=True, padx=20, pady=20)
        # Scroll to browse frames in preview
        self.preview_label.bind("<MouseWheel>", self._on_preview_scroll)
        self.preview_label.bind("<Button-4>", self._on_preview_scroll)
        self.preview_label.bind("<Button-5>", self._on_preview_scroll)

        # Stats row
        stats_frame = tk.Frame(right_panel, bg=PANEL_COLOR)
        stats_frame.pack(fill="x", pady=(0, 4))
        for _ in range(4):
            stats_frame.grid_columnconfigure(_, weight=1)
        self._add_stat(stats_frame, 0, self.stat_events, "检测事件")
        self._add_stat(stats_frame, 1, self.stat_keyframes, "关键帧")
        self._add_stat(stats_frame, 2, self.stat_time, "处理耗时")
        self._add_stat(stats_frame, 3, self.stat_output, "输出路径")

        # Log panel (collapsible)
        log_header = tk.Frame(right_panel, bg=PANEL_COLOR, cursor="hand2")
        log_header.pack(fill="x", ipady=4)
        log_header.bind("<Button-1>", self._toggle_log)
        self.log_toggle_label = tk.Label(log_header, text="📋 运行日志  ▲",
                                          bg=PANEL_COLOR, fg=MUTED_COLOR,
                                          font=("Microsoft YaHei", 9, "bold"))
        self.log_toggle_label.pack(anchor="w", padx=12)
        self.log_text = tk.Text(right_panel, bg="#fafbfc", fg=TEXT_COLOR,
                                 font=("Consolas", 9), state="disabled", wrap="word",
                                 height=8, bd=0, padx=8, pady=4)
        self.log_text.pack(fill="x", pady=(0, 0))

        # ── Bottom Status Bar ──
        bottombar = tk.Frame(self, bg=PANEL_COLOR, height=28, bd=0, highlightthickness=0)
        bottombar.pack(fill="x", side="bottom")
        bottombar.pack_propagate(False)
        status_dot = tk.Label(bottombar, text="●", bg=PANEL_COLOR, fg=SUCCESS_COLOR, font=("", 10))
        status_dot.pack(side="left", padx=(12, 4))
        tk.Label(bottombar, textvariable=self.status_text, bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("Microsoft YaHei", 9)).pack(side="left")
        tk.Label(bottombar, textvariable=self.status_extra, bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 9)).pack(side="right", padx=12)

    def _create_step_card(self, parent, step_id: str, name: str) -> tk.Frame:
        card = tk.Frame(parent, bg=PANEL_COLOR, bd=1, relief="solid",
                         highlightbackground=BORDER_COLOR, highlightcolor=BORDER_COLOR, highlightthickness=1)
        card.pack(fill="x", padx=10, pady=4)
        header = tk.Frame(card, bg=PANEL_COLOR)
        header.pack(fill="x", padx=10, pady=(8, 2))
        num_label = tk.Label(header, text=step_id, bg=BORDER_COLOR, fg=MUTED_COLOR,
                             font=("Microsoft YaHei", 11, "bold"), width=3, height=1)
        num_label.pack(side="left", padx=(0, 8))
        tk.Label(header, text=name, bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        desc_label = tk.Label(card, textvariable=self.step_desc[step_id], bg=PANEL_COLOR, fg=MUTED_COLOR,
                              font=("Microsoft YaHei", 8), anchor="w")
        desc_label.pack(fill="x", padx=10, pady=(0, 4))
        progress = ttk.Progressbar(card, variable=self.step_progress[step_id], maximum=100)
        progress.pack(fill="x", padx=10, pady=(0, 8))
        # Store references for state updates
        card.num_label = num_label
        card.progress_bar = progress
        return card

    def _add_stat(self, parent, col, var, label):
        f = tk.Frame(parent, bg=PANEL_COLOR)
        f.grid(row=0, column=col, sticky="ew", padx=12, pady=10)
        tk.Label(f, textvariable=var, bg=PANEL_COLOR, fg=ACCENT_COLOR,
                 font=("Microsoft YaHei", 16, "bold")).pack()
        tk.Label(f, text=label, bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 8)).pack()

    def _set_step_state(self, step_id: str, state: str, desc: str = ""):
        self.step_state[step_id] = state
        card = self.step_frames[step_id]
        if state == "done":
            card.num_label.config(bg=SUCCESS_COLOR, fg="white", text="✓")
            card.config(highlightbackground=SUCCESS_COLOR, highlightcolor=SUCCESS_COLOR)
        elif state == "active":
            card.num_label.config(bg=ACCENT_COLOR, fg="white", text=step_id)
            card.config(highlightbackground=ACCENT_COLOR, highlightcolor=ACCENT_COLOR)
        else:
            card.num_label.config(bg=BORDER_COLOR, fg=MUTED_COLOR, text=step_id)
            card.config(highlightbackground=BORDER_COLOR, highlightcolor=BORDER_COLOR)
        if desc:
            self.step_desc[step_id].set(desc)

    def _toggle_log(self, event=None):
        if self.log_expanded.get():
            self.log_text.pack_forget()
            self.log_toggle_label.config(text="📋 运行日志  ▶")
        else:
            self.log_text.pack(fill="x", pady=(0, 0), after=self.log_toggle_label.master)
            self.log_toggle_label.config(text="📋 运行日志  ▲")
        self.log_expanded.set(not self.log_expanded.get())

    def _select_video(self):
        filename = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.flv *.wmv"), ("所有文件", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
            logging.info("已选择视频: %s", filename)
            try:
                self.current_metadata = get_video_metadata(filename)
                duration = self.current_metadata.frame_count / max(self.current_metadata.fps, 1)
                self.stat_output.set(Path(self.app_config.output_path).name)
                self.status_extra.set(f"输出: {self.app_config.output_path}")
                self.start_btn.config(state="normal")
                self._update_preview_hint()
            except Exception as e:
                logging.error("读取视频失败: %s", e)
                messagebox.showerror("错误", f"无法读取视频: {e}")

    def _update_preview_hint(self):
        if self.current_metadata:
            m = self.current_metadata
            self.preview_label.config(
                text=f"已加载: {Path(m.path).name}\n{m.width}x{m.height} | {m.fps:.1f} fps | {m.frame_count} 帧\n点击"▶ 开始处理"启动分析",
                fg=TEXT_COLOR)

    def _start_processing(self):
        if not self.video_path.get():
            messagebox.showwarning("警告", "请先选择视频文件")
            return
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            messagebox.showwarning("警告", "请等待当前处理完成")
            return
        # Reset state
        for sid in ["1", "2", "3", "4"]:
            self._set_step_state(sid, "waiting", "等待开始")
            self.step_progress[sid].set(0)
        self.stat_events.set("--")
        self.stat_keyframes.set("--")
        self.stat_time.set("--")
        self.start_btn.config(state="disabled")
        self._set_step_state("1", "active", "分解中...")
        import time as _time
        self._pipeline_start = _time.time()

        def worker():
            try:
                self.ui_queue.put(("step", {"id": "1", "state": "active", "desc": "分解中..."}))
                # Step 1: Extract frames
                temp_dir = self.app_config.temp_path or "temp/frames"
                metadata = video_to_images(
                    self.video_path.get(), temp_dir, self.proc_config.sample_stride,
                    progress_callback=lambda **kw: self.ui_queue.put(("progress", {"step": "1", **kw})),
                    max_width=self.proc_config.max_width, max_height=self.proc_config.max_height,
                )
                self.current_metadata = metadata
                self.source_sampled_frames = list_image_files(Path(temp_dir))
                self.ui_queue.put(("step", {"id": "1", "state": "done", "desc": f"提取 {len(self.source_sampled_frames)} 帧"}))

                # Step 2: Intrusion detection
                self.ui_queue.put(("step", {"id": "2", "state": "active", "desc": "检测中..."}))
                events = detect_intrusion_events(
                    self.video_path.get(), sampling_stride=2,
                    min_event_seconds=self.proc_config.min_event_duration,
                    motion_weight=self.proc_config.motion_weight,
                    person_weight=self.proc_config.person_weight,
                    intrusion_threshold=self.proc_config.intrusion_threshold,
                    progress_callback=lambda **kw: self.ui_queue.put(("progress", {"step": "2", **kw})),
                )
                self.current_intrusion_events = events
                self.ui_queue.put(("step", {"id": "2", "state": "done", "desc": f"检测到 {len(events)} 个事件"}))

                # Step 3: Keyframe selection
                self.ui_queue.put(("step", {"id": "3", "state": "active", "desc": "识别中..."}))
                allowed = map_events_to_image_files(self.source_sampled_frames, events)
                result = auto_select_keyframes_by_clustering(
                    temp_dir, n_clusters=self.proc_config.keyframe_count,
                    progress_callback=lambda **kw: self.ui_queue.put(("progress", {"step": "3", **kw})),
                    metadata=metadata, allowed_images=allowed,
                    intrusion_events=events, sampling_stride=self.proc_config.sample_stride,
                )
                self.analysis_result = result
                self.current_selected_frames = result.recommended_keyframes
                self.ui_queue.put(("step", {"id": "3", "state": "done", "desc": f"选出 {len(result.recommended_keyframes)} 个关键帧"}))

                # Step 4: Summary (actual export is manual)
                self.ui_queue.put(("step", {"id": "4", "state": "done", "desc": "处理完成，可导出"}))
                self.ui_queue.put(("stats", {
                    "events": str(len(events)),
                    "keyframes": str(len(result.recommended_keyframes)),
                    "time": format_seconds(_time.time() - self._pipeline_start),
                }))
                self.ui_queue.put(("done", {}))
            except Exception as e:
                logging.error("流水线执行失败: %s", e)
                self.ui_queue.put(("error", {"msg": str(e)}))

        self.pipeline_thread = threading.Thread(target=worker, daemon=True)
        self.pipeline_thread.start()

    def handle_ui_action(self, action: str, data: dict):
        """Called by the app's poll_queues to process UI updates from bg threads."""
        if action == "step":
            self._set_step_state(data["id"], data["state"], data.get("desc", ""))
        elif action == "progress":
            sid = data.get("step", "1")
            current = data.get("current", 0)
            total = data.get("total", 1)
            pct = (current / total * 100) if total > 0 else 0
            self.step_progress[sid].set(pct)
            if data.get("saved_count"):
                self._set_step_state("1", "active", f"已保存 {data['saved_count']} 帧")
            if data.get("found_count") is not None and sid == "2":
                self._set_step_state("2", "active", f"找到 {data['found_count']} 个事件")
            self._update_preview_if_available()
        elif action == "stats":
            self.stat_events.set(data.get("events", "--"))
            self.stat_keyframes.set(data.get("keyframes", "--"))
            self.stat_time.set(data.get("time", "--"))
        elif action == "done":
            self.status_text.set("处理完成")
            self.start_btn.config(state="normal")
            self._update_preview_if_available()
        elif action == "error":
            messagebox.showerror("错误", f"处理失败: {data.get('msg', '')}")
            self.status_text.set("处理失败")
            self.start_btn.config(state="normal")

    def _update_preview_if_available(self):
        """Show first keyframe/highlight frame in preview area."""
        if self.current_selected_frames:
            temp_dir = Path(self.app_config.temp_path or "temp/frames")
            filepath = temp_dir / self.current_selected_frames[0]
            if filepath.exists():
                try:
                    img = Image.open(filepath)
                    w = self.preview_label.winfo_width()
                    h = self.preview_label.winfo_height()
                    if w > 1 and h > 1:
                        r = min(w / img.width, h / img.height, 1.0)
                        img = img.resize((int(img.width * r), int(img.height * r)), Image.Resampling.LANCZOS)
                    self.preview_photo = ImageTk.PhotoImage(img)
                    self.preview_label.config(image=self.preview_photo, text="")
                except Exception:
                    pass

    def _on_preview_scroll(self, event):
        if not self.current_selected_frames:
            return
        delta = 1 if (hasattr(event, 'delta') and event.delta > 0) or event.num == 4 else -1
        self._preview_index = max(0, min(len(self.current_selected_frames) - 1,
                                          self._preview_index + delta))
        temp_dir = Path(self.app_config.temp_path or "temp/frames")
        filepath = temp_dir / self.current_selected_frames[self._preview_index]
        if filepath.exists():
            try:
                img = Image.open(filepath)
                w = self.preview_label.winfo_width()
                h = self.preview_label.winfo_height()
                if w > 1 and h > 1:
                    r = min(w / img.width, h / img.height, 1.0)
                    img = img.resize((int(img.width * r), int(img.height * r)), Image.Resampling.LANCZOS)
                self.preview_photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=self.preview_photo, text="")
            except Exception:
                pass

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.after(PROGRESS_POLL_MS, self._poll_log)

    def refresh_from_config(self):
        """Called when returning from settings page."""
        self.stat_output.set(Path(self.app_config.output_path).name)
        self.status_extra.set(f"输出: {self.app_config.output_path}")
```

- [ ] **Step 2: Commit**

```bash
git add src/gui_main.py
git commit -m "feat: add redesigned main page with step cards, stats, collapsible log"
```

---

### Task 9: Create `src/gui_app.py`

**Files:** Create `src/gui_app.py`

Application root: window management, page switching, queue polling, config init.

- [ ] **Step 1: Write `src/gui_app.py`**

```python
"""Application framework — window, page switching, queue polling."""
import logging
import queue
import tkinter as tk
from pathlib import Path

from src.config import (AppConfig, ProcessingConfig, SURFACE_COLOR, APP_DIR,
                        CONFIG_DIR, LOG_DIR, TEMP_DIR, OUTPUT_DIR, PROGRESS_POLL_MS)
from src.gui_main import MainPage
from src.gui_settings import SettingsPage
from src.utils import setup_logging

APP_TITLE = "视频关键帧筛选与重构系统"


class VideoProcessorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1200x780")
        self.root.minsize(1024, 640)
        self.root.configure(bg=SURFACE_COLOR)

        # Config
        self.app_config = AppConfig()
        # Reload from disk
        loaded = AppConfig(
            output_path=str(OUTPUT_DIR),
            temp_path=str(TEMP_DIR),
            sample_stride=self.app_config.sample_stride,
            intrusion_threshold=self.app_config.intrusion_threshold,
            min_event_duration=self.app_config.min_event_duration,
            motion_weight=self.app_config.motion_weight,
            person_weight=self.app_config.person_weight,
            max_width=self.app_config.max_width,
        )
        # Try loading saved config
        from src.config import load_app_config, save_app_config
        self.app_config = load_app_config()
        # Ensure directories
        if self.app_config.remember_last_output:
            Path(self.app_config.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.app_config.temp_path).mkdir(parents=True, exist_ok=True)

        # Processing config (runtime values, not persisted separately)
        self.proc_config = ProcessingConfig(
            sample_stride=self.app_config.sample_stride,
            intrusion_threshold=self.app_config.intrusion_threshold,
            min_event_duration=self.app_config.min_event_duration,
            motion_weight=self.app_config.motion_weight,
            person_weight=self.app_config.person_weight,
            max_width=self.app_config.max_width,
        )

        # Logging
        self.log_queue: queue.Queue = queue.Queue()
        self.ui_queue: queue.Queue = queue.Queue()
        self.gui_handler, self.file_handler = setup_logging(LOG_DIR, self.log_queue)

        # Pages
        self.main_page = MainPage(
            self.root, self.app_config, self.proc_config,
            self.log_queue, self.ui_queue, self._show_settings,
        )
        self.settings_page = SettingsPage(
            self.root, self.app_config, self._show_main, self._on_settings_saved,
        )
        self._show_main()
        self._poll_ui()

        logging.info("应用程序初始化完成")

    def _show_main(self):
        self.settings_page.pack_forget()
        self.main_page.pack(fill="both", expand=True)

    def _show_settings(self):
        self.main_page.pack_forget()
        self.settings_page._load_config()
        self.settings_page.pack(fill="both", expand=True)

    def _on_settings_saved(self):
        # Sync processing config from app config
        c = self.app_config
        self.proc_config.sample_stride = c.sample_stride
        self.proc_config.intrusion_threshold = c.intrusion_threshold
        self.proc_config.min_event_duration = c.min_event_duration
        self.proc_config.motion_weight = c.motion_weight
        self.proc_config.person_weight = c.person_weight
        self.proc_config.max_width = c.max_width
        self.main_page.refresh_from_config()
        self._show_main()
        logging.info("设置已保存")

    def _poll_ui(self):
        try:
            while True:
                action, data = self.ui_queue.get_nowait()
                self.main_page.handle_ui_action(action, data)
        except queue.Empty:
            pass
        self.root.after(PROGRESS_POLL_MS, self._poll_ui)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = VideoProcessorApp()
    app.run()
```

- [ ] **Step 2: Commit**

```bash
git add src/gui_app.py
git commit -m "feat: add application framework with page switching and queue polling"
```

---

### Task 10: Add `requirements.txt` and `.gitignore`, remove old `src/main.py`

**Files:** Create `requirements.txt`, `.gitignore`; Delete `src/main.py`

- [ ] **Step 1: Create `requirements.txt`**

```
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=9.0.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.0.0
imageio-ffmpeg>=0.4.0
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.vscode/
.idea/
temp/
output/
logs/
*.log
.superpowers/
.env
```

- [ ] **Step 3: Delete old file and commit**

```bash
git rm src/main.py
git add requirements.txt .gitignore
git commit -m "chore: add requirements.txt, .gitignore; remove old monolithic main.py"
```

---

### Task 11: Verify application runs

- [ ] **Step 1: Run the application to verify imports work**

```bash
cd D:/All_Python/Project/torch_env_3 && python -c "from src.config import AppConfig; print('config OK')" && python -c "from src.utils import sanitize_name; print('utils OK')" && python -c "from src.detection import resize_for_analysis; print('detection OK')" && python -c "from src.video_io import get_video_metadata; print('video_io OK')" && python -c "from src.gui_settings import SettingsPage; print('gui_settings OK')" && python -c "from src.gui_main import MainPage; print('gui_main OK')" && python -c "from src.gui_app import VideoProcessorApp; print('gui_app OK')"
```

Expected: All modules import successfully (or skip features.py/keyframes.py if torch unavailable).

- [ ] **Step 2: Commit any fixes**

```bash
git add -A && git commit -m "fix: module import fixes after verification"
```

---

### Task 12: Final integration test

- [ ] **Step 1: Launch the full GUI and verify pages switch**

Launch `python src/gui_app.py`. Verify:
- Main page shows with step cards, stats row, log panel
- "⚙ 设置" button opens settings page
- "← 返回" button returns to main page
- "保存设置" saves config to disk
- Settings persist across app restart

- [ ] **Step 2: Commit final fixes**

```bash
git add -A && git commit -m "fix: final integration fixes"
```
