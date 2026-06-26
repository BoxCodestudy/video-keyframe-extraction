# Camera Recording + UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live camera recording as a video source, modernize the UI to a clean professional theme, relocate progress bars to a unified bottom panel, add auto-export and camera-save-path settings, and sync docs.

**Architecture:** A new `camera_recorder.py` module handles OpenCV capture + Tkinter recording dialog. `config.py` gets updated color constants and two new `AppConfig` fields. `gui_main.py` undergoes the largest refactor: dropdown video-source menu, simplified left step panel, unified bottom progress bar. `gui_settings.py` gains two new setting rows. `gui_app.py` gets minor wiring updates. All backend modules (`detection.py`, `features.py`, `keyframes.py`, `video_io.py`) remain unchanged.

**Tech Stack:** Python 3.8+, Tkinter, OpenCV (cv2), Pillow, NumPy

---

### Task 1: Update color theme and add new config fields

**Files:**
- Modify: `src/config.py:29-39` (theme constants)
- Modify: `src/config.py:57-68` (AppConfig dataclass)

- [ ] **Step 1: Replace color theme constants**

Replace lines 30-38 in `src/config.py`:

```python
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
```

Old values to replace:
- `ACCENT_COLOR`: `#1a73e8` → `#2563eb`
- `ACCENT_HOVER`: `#1557b0` → `#1d4ed8`
- `SUCCESS_COLOR`: `#0ea854` → `#16a34a`
- `WARNING_COLOR`: `#e37400` → `#d97706`
- `SURFACE_COLOR`: `#f0f2f5` → `#f5f7fa`
- `TEXT_COLOR`: `#333333` → `#1e293b`
- `MUTED_COLOR`: `#888888` → `#64748b`
- `BORDER_COLOR`: `#e0e4e8` → `#e2e8f0`
- Add `DANGER_COLOR` = `#ef4444`

- [ ] **Step 2: Add new fields to AppConfig**

In `src/config.py`, add two new fields to the `AppConfig` dataclass (after line 68, before `__post_init__`):

```python
camera_save_path: str = ""
auto_export_video: bool = True
```

Full updated AppConfig:

```python
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
```

- [ ] **Step 3: Verify config loads correctly**

Run:
```bash
cd d:/All_Python/Project/torch_env_3 && python -c "from src.config import AppConfig, ACCENT_COLOR, DANGER_COLOR; c = AppConfig(); print(f'Colors: {ACCENT_COLOR} {DANGER_COLOR}'); print(f'Camera path: {c.camera_save_path}'); print(f'Auto export: {c.auto_export_video}')"
```
Expected: prints `#2563eb #ef4444`, camera path ending in `saved_video`, `True`.

- [ ] **Step 4: Commit**

```bash
cd d:/All_Python/Project/torch_env_3 && git add src/config.py && git commit -m "feat: update color theme and add camera_save_path, auto_export_video config fields"
```

---

### Task 2: Create camera recorder module

**Files:**
- Create: `src/camera_recorder.py`

- [ ] **Step 1: Write the camera recorder module**

Create `src/camera_recorder.py`:

```python
"""Camera recording — OpenCV capture with Tkinter recording dialog."""
import logging
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
from PIL import Image, ImageTk

# These will be imported from config at runtime,
# but defined locally here so this module can be tested standalone.
try:
    from src.config import DANGER_COLOR, ACCENT_COLOR, SURFACE_COLOR, PANEL_COLOR, TEXT_COLOR, MUTED_COLOR, BORDER_COLOR
except ImportError:
    DANGER_COLOR = "#ef4444"
    ACCENT_COLOR = "#2563eb"
    SURFACE_COLOR = "#f5f7fa"
    PANEL_COLOR = "#ffffff"
    TEXT_COLOR = "#1e293b"
    MUTED_COLOR = "#64748b"
    BORDER_COLOR = "#e2e8f0"

RECORDING_FPS = 20.0
RECORDING_CODEC = "mp4v"


class CameraRecorder:
    """Manage camera capture and MP4 encoding."""

    def __init__(self, camera_index: int = 0, fps: float = RECORDING_FPS):
        self.camera_index = camera_index
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._output_path: str = ""
        self._frame_width = 640
        self._frame_height = 480

    @staticmethod
    def list_cameras(max_check: int = 4) -> list[dict]:
        """Enumerate available camera devices."""
        cameras = []
        for idx in range(max_check):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({"index": idx, "width": w, "height": h, "name": f"Camera {idx}"})
                cap.release()
        return cameras

    def open(self, camera_index: int = 0) -> bool:
        """Open a camera device. Returns True on success."""
        self.camera_index = camera_index
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            logging.warning("无法打开摄像头 index=%d", camera_index)
            return False
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("摄像头已打开: index=%d, %dx%d", camera_index, self._frame_width, self._frame_height)
        return True

    def get_frame(self) -> Optional["np.ndarray"]:
        """Read one frame. Returns None if capture fails."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def start_recording(self, save_dir: str) -> str:
        """Begin writing video to save_dir. Returns the output file path."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{stamp}.mp4"
        self._output_path = str(Path(save_dir) / filename)
        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        self._writer = cv2.VideoWriter(
            self._output_path, fourcc, self.fps,
            (self._frame_width, self._frame_height),
        )
        self._recording = True
        logging.info("开始录制: %s", self._output_path)
        return self._output_path

    def write_frame(self, frame) -> bool:
        """Write a frame to the recording. Frame should be RGB numpy array."""
        if not self._recording or self._writer is None:
            return False
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)
        return True

    def stop_recording(self) -> str:
        """Stop recording and release the writer. Returns output path."""
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logging.info("录制已停止: %s", self._output_path)
        return self._output_path

    def is_recording(self) -> bool:
        return self._recording

    def release(self):
        """Release all resources."""
        if self._recording:
            self.stop_recording()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logging.info("摄像头资源已释放")


class CameraDialog(tk.Toplevel):
    """Camera recording popup window with live preview and record controls."""

    def __init__(self, parent, save_dir: str, on_video_ready: Callable[[str], None],
                 camera_index: int = 0):
        super().__init__(parent)
        self.title("摄像头录制")
        self.geometry("720x560")
        self.resizable(False, False)
        self.configure(bg="#0f172a")
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.save_dir = save_dir
        self.on_video_ready = on_video_ready
        self._recorder = CameraRecorder(camera_index=camera_index, fps=RECORDING_FPS)
        self._elapsed_seconds = 0
        self._timer_id: Optional[str] = None
        self._preview_photo = None

        self._build_ui()

        if not self._recorder.open(camera_index):
            self._status_label.config(text="⚠ 无法打开摄像头", fg="#ef4444")
        else:
            self._status_label.config(text="● 摄像头就绪", fg="#22c55e")
            self._start_preview_loop()

    def _build_ui(self):
        # ── Title bar ──
        title_frame = tk.Frame(self, bg="#0f172a")
        title_frame.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(title_frame, text="🎥 摄像头录制", bg="#0f172a", fg="white",
                 font=("Microsoft YaHei", 14, "bold")).pack(side="left")
        self._status_label = tk.Label(title_frame, text="● 初始化中...", bg="#0f172a", fg="#fbbf24",
                                       font=("Microsoft YaHei", 10))
        self._status_label.pack(side="right")

        # ── Preview area ──
        preview_frame = tk.Frame(self, bg="#1e293b", bd=2, relief="solid",
                                  highlightbackground="#334155", highlightthickness=1)
        preview_frame.pack(fill="both", expand=True, padx=16, pady=8)
        self._preview_label = tk.Label(preview_frame, bg="#1e293b", anchor="center",
                                        text="📹\n摄像头预览区域", fg="#475569",
                                        font=("Microsoft YaHei", 14))
        self._preview_label.pack(fill="both", expand=True)

        # Recording indicator (overlaid concept — just a label for Tkinter)
        self._rec_indicator = tk.Label(preview_frame, text="", bg="#1e293b", fg="#ef4444",
                                        font=("Microsoft YaHei", 11, "bold"))
        self._rec_indicator.place(relx=1.0, x=-12, y=8, anchor="ne")

        # ── Controls ──
        ctrl_frame = tk.Frame(self, bg="#0f172a")
        ctrl_frame.pack(fill="x", padx=16, pady=(8, 4))
        btn_frame = tk.Frame(ctrl_frame, bg="#0f172a")
        btn_frame.pack()
        self._rec_btn = tk.Button(btn_frame, text="🔴 开始录制", bg=DANGER_COLOR, fg="white",
                                   font=("Microsoft YaHei", 11, "bold"), bd=0,
                                   padx=20, pady=8, cursor="hand2", command=self._toggle_recording,
                                   activebackground="#dc2626")
        self._rec_btn.pack(side="left", padx=6)
        self._cancel_btn = tk.Button(btn_frame, text="✕ 取消", bg="#334155", fg="#94a3b8",
                                      font=("Microsoft YaHei", 11), bd=0,
                                      padx=20, pady=8, cursor="hand2", command=self._on_cancel,
                                      activebackground="#475569")
        self._cancel_btn.pack(side="left", padx=6)

        # ── Info row ──
        info_frame = tk.Frame(self, bg="#0f172a")
        info_frame.pack(fill="x", padx=16, pady=(4, 12))
        self._info_label = tk.Label(info_frame, text=f"📐 {self._recorder._frame_width}×{self._recorder._frame_height}  ⚡ {RECORDING_FPS:.0f} fps  💾 {self.save_dir}  📝 MP4",
                                     bg="#0f172a", fg="#64748b", font=("Consolas", 9))
        self._info_label.pack(side="left")

    def _start_preview_loop(self):
        """Poll camera frames and update the preview label."""
        frame = self._recorder.get_frame()
        if frame is not None:
            if self._recorder.is_recording():
                self._recorder.write_frame(frame)
            # Resize for preview display
            h, w = frame.shape[:2]
            target_w = 680
            scale = target_w / max(w, 1)
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            img = Image.fromarray(frame)
            self._preview_photo = ImageTk.PhotoImage(img)
            self._preview_label.config(image=self._preview_photo, text="")
        self._timer_id = self.after(50, self._start_preview_loop)

    def _toggle_recording(self):
        if not self._recorder.is_recording():
            self._recorder.start_recording(self.save_dir)
            self._elapsed_seconds = 0
            self._rec_btn.config(text="⏹ 停止录制", bg="#334155", fg="white",
                                 activebackground="#475569")
            self._rec_indicator.config(text="● 录制中 00:00")
            self._update_recording_timer()
        else:
            self._stop_and_return()

    def _update_recording_timer(self):
        if not self._recorder.is_recording():
            return
        self._elapsed_seconds += 1
        mm = self._elapsed_seconds // 60
        ss = self._elapsed_seconds % 60
        self._rec_indicator.config(text=f"● 录制中 {mm:02d}:{ss:02d}")
        self.after(1000, self._update_recording_timer)

    def _stop_and_return(self):
        output_path = self._recorder.stop_recording()
        self._recorder.release()
        if self._timer_id:
            self.after_cancel(self._timer_id)
        self.destroy()
        if output_path and Path(output_path).exists():
            logging.info("录制完成，视频保存至: %s", output_path)
            self.on_video_ready(output_path)

    def _on_cancel(self):
        if self._recorder.is_recording():
            self._recorder.stop_recording()
        self._recorder.release()
        if self._timer_id:
            self.after_cancel(self._timer_id)
        self.destroy()
        logging.info("摄像头录制已取消")
```

- [ ] **Step 2: Verify the module imports cleanly**

Run:
```bash
cd d:/All_Python/Project/torch_env_3 && python -c "from src.camera_recorder import CameraRecorder, CameraDialog; cams = CameraRecorder.list_cameras(); print(f'Found {len(cams)} camera(s)'); print(cams)"
```
Expected: prints number of cameras found (may be 0 if no camera, or 1+ if webcam present).

- [ ] **Step 3: Commit**

```bash
cd d:/All_Python/Project/torch_env_3 && git add src/camera_recorder.py && git commit -m "feat: add camera recorder module with recording dialog"
```

---

### Task 3: Refactor gui_main.py — modern UI layout

**Files:**
- Modify: `src/gui_main.py` — full rewrite of `_build_ui()` and related methods

This is the largest task. Key changes:
1. Replace top bar: remove independent select/start/export buttons, add dropdown menu
2. Simplify left panel: step cards become indicators (no progress bars)
3. Add bottom unified progress bar panel
4. Update all color references

- [ ] **Step 1: Update imports in gui_main.py**

Replace the import line at the top of `gui_main.py` (line 14-17):

```python
from src.config import (AppConfig, ProcessingConfig, AnalysisResult, VideoMetadata,
                        IntrusionEvent, SURFACE_COLOR, PANEL_COLOR, TEXT_COLOR,
                        MUTED_COLOR, ACCENT_COLOR, SUCCESS_COLOR, WARNING_COLOR,
                        BORDER_COLOR, PROGRESS_POLL_MS, DANGER_COLOR)
```

Old line includes the same import names — just add `DANGER_COLOR`.

- [ ] **Step 2: Add camera_recorder import**

After the existing imports (after line 23), add:

```python
from src.camera_recorder import CameraDialog
```

- [ ] **Step 3: Replace `_build_ui` method — top bar with dropdown menu**

Replace lines 66-91 (the entire `_build_ui` topbar section through the export_btn line):

```python
    def _build_ui(self):
        # ── Top Bar ──
        topbar = tk.Frame(self, bg=PANEL_COLOR, height=52, bd=0, highlightthickness=0)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="◈ 基于深度特征聚类的视频关键帧提取与事件检测系统",
                 bg=PANEL_COLOR, fg=ACCENT_COLOR,
                 font=("Microsoft YaHei", 13, "bold")).pack(side="left", padx=16, pady=12)
        btn_frame = tk.Frame(topbar, bg=PANEL_COLOR)
        btn_frame.pack(side="right", padx=12, pady=10)

        # ── Video source dropdown (Menubutton) ──
        self.src_menu_btn = tk.Menubutton(btn_frame, text="📁 视频来源 ▾",
                                           bg=PANEL_COLOR, fg=TEXT_COLOR,
                                           font=("Microsoft YaHei", 10),
                                           bd=1, relief="solid", padx=10, pady=4,
                                           cursor="hand2", activebackground="#f1f5f9",
                                           activeforeground=TEXT_COLOR)
        src_menu = tk.Menu(self.src_menu_btn, tearoff=0, bg=PANEL_COLOR, fg=TEXT_COLOR,
                           font=("Microsoft YaHei", 10), activebackground="#eff6ff",
                           activeforeground=ACCENT_COLOR)
        src_menu.add_command(label="📂 从文件选择", command=self._select_video)
        src_menu.add_command(label="🎥 从摄像头录制", command=self._open_camera)
        self.src_menu_btn.config(menu=src_menu)
        self.src_menu_btn.pack(side="left", padx=4)

        self.settings_btn = tk.Button(btn_frame, text="⚙ 设置", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                      font=("Microsoft YaHei", 10), bd=1, relief="solid",
                                      padx=10, pady=4, cursor="hand2",
                                      activebackground="#f1f5f9", command=self.on_open_settings)
        self.settings_btn.pack(side="left", padx=4)
        self.start_btn = tk.Button(btn_frame, text="▶ 开始处理", bg=ACCENT_COLOR, fg="white",
                                   font=("Microsoft YaHei", 10, "bold"), bd=0,
                                   padx=14, pady=4, cursor="hand2",
                                   activebackground=ACCENT_COLOR,
                                   command=self._start_processing, state="disabled")
        self.start_btn.pack(side="left", padx=4)
        self.export_btn = tk.Button(btn_frame, text="📤 导出", bg=SUCCESS_COLOR, fg="white",
                                     font=("Microsoft YaHei", 10, "bold"), bd=0,
                                     padx=14, pady=4, cursor="hand2",
                                     activebackground=SUCCESS_COLOR,
                                     command=self._start_export, state="disabled")
        self.export_btn.pack(side="left", padx=4)
```

- [ ] **Step 4: Replace left panel — simplified step indicators**

Replace lines 93-107 (left panel section through `_create_step_card`).

First, the left panel section (replace lines 97-107):

```python
        # ── Main Content ──
        main = tk.Frame(self, bg=SURFACE_COLOR)
        main.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # Left: Step indicators (simplified, no progress bars)
        left_panel = tk.Frame(main, bg=PANEL_COLOR, width=190)
        left_panel.pack(side="left", fill="y", padx=(0, 8))
        left_panel.pack_propagate(False)
        tk.Label(left_panel, text="处理步骤", bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 9, "bold")).pack(anchor="w", padx=12, pady=(12, 8))
        step_names = ["视频分解与采样", "入侵检测", "关键帧筛选", "视频重构"]
        self.step_frames = {}
        for i, name in enumerate(step_names, 1):
            self.step_frames[str(i)] = self._create_step_indicator(left_panel, str(i), name)
```

- [ ] **Step 5: Replace `_create_step_card` with new `_create_step_indicator` method**

Replace lines 154-171 (the `_create_step_card` method):

```python
    def _create_step_indicator(self, parent, step_id: str, name: str) -> tk.Frame:
        """Create a compact step indicator (no progress bar)."""
        frame = tk.Frame(parent, bg=PANEL_COLOR)
        frame.pack(fill="x", padx=10, pady=3)

        inner = tk.Frame(frame, bg=PANEL_COLOR, bd=0, highlightthickness=0)
        inner.pack(fill="x", padx=8, pady=6)

        num_label = tk.Label(inner, text=step_id, bg=BORDER_COLOR, fg="white",
                             font=("Microsoft YaHei", 10, "bold"), width=3, height=1,
                             bd=0)
        num_label.pack(side="left", padx=(0, 10))
        name_label = tk.Label(inner, text=name, bg=PANEL_COLOR, fg=TEXT_COLOR,
                              font=("Microsoft YaHei", 10, "bold"), anchor="w")
        name_label.pack(side="left")
        desc_label = tk.Label(frame, textvariable=self.step_desc[step_id], bg=PANEL_COLOR,
                               fg=MUTED_COLOR, font=("Microsoft YaHei", 8), anchor="w")
        desc_label.pack(fill="x", padx=(50, 10), pady=(0, 6))

        frame.num_label = num_label
        frame.name_label = name_label
        frame.desc_label = desc_label
        # Thin left border simulated via a colored edge bar
        frame.edge = tk.Frame(frame, bg=BORDER_COLOR, width=3)
        frame.edge.pack(side="left", fill="y", before=inner)
        frame.edge.pack_configure(before=inner)
        return frame
```

Wait — the `.pack_configure(before=...)` approach in Tkinter won't create a left border properly. Let me redesign the step indicator with a proper left-border approach:

```python
    def _create_step_indicator(self, parent, step_id: str, name: str) -> tk.Frame:
        """Create a compact step indicator with left color bar."""
        outer = tk.Frame(parent, bg=BORDER_COLOR, bd=0, highlightthickness=0)
        outer.pack(fill="x", padx=10, pady=3)
        inner = tk.Frame(outer, bg=PANEL_COLOR, bd=0)
        inner.pack(fill="x", padx=(3, 0), pady=0)

        top_row = tk.Frame(inner, bg=PANEL_COLOR)
        top_row.pack(fill="x", padx=8, pady=(6, 2))
        num_label = tk.Label(top_row, text=step_id, bg=BORDER_COLOR, fg="white",
                             font=("Microsoft YaHei", 10, "bold"), width=3, height=1)
        num_label.pack(side="left", padx=(0, 10))
        tk.Label(top_row, text=name, bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        desc_label = tk.Label(inner, textvariable=self.step_desc[step_id], bg=PANEL_COLOR,
                              fg=MUTED_COLOR, font=("Microsoft YaHei", 8), anchor="w")
        desc_label.pack(fill="x", padx=(44, 8), pady=(0, 6))

        outer.num_label = num_label
        outer.desc_label = desc_label
        return outer
```

- [ ] **Step 6: Add bottom unified progress bar panel**

After the stats_frame section and before the log section, add a unified progress bar.

Find lines 122-141 (the stats_frame through log_text section) and replace the area AFTER the log_text:

Add after line 141 (after `self.log_text.pack(fill="x")`):

```python

        # ── Unified Progress Bar Panel (bottom of right panel) ──
        self.progress_panel = tk.Frame(right_panel, bg=PANEL_COLOR, bd=1, relief="solid",
                                        highlightbackground=BORDER_COLOR, highlightthickness=0)
        self.progress_panel.pack(fill="x", pady=(4, 0), ipady=8)

        progress_inner = tk.Frame(self.progress_panel, bg=PANEL_COLOR)
        progress_inner.pack(fill="x", padx=14, pady=8)

        self.progress_step_badge = tk.Label(progress_inner, text="—", bg=BORDER_COLOR, fg="white",
                                             font=("Microsoft YaHei", 10, "bold"), width=3, height=1)
        self.progress_step_badge.pack(side="left", padx=(0, 10))

        self.progress_label = tk.Label(progress_inner, text="就绪", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                        font=("Microsoft YaHei", 10, "bold"))
        self.progress_label.pack(side="left")
        self.progress_step_hint = tk.Label(progress_inner, text="", bg=PANEL_COLOR, fg=MUTED_COLOR,
                                            font=("Microsoft YaHei", 9))
        self.progress_step_hint.pack(side="left", padx=(8, 0))

        self.progress_pct = tk.Label(progress_inner, text="", bg=PANEL_COLOR, fg=ACCENT_COLOR,
                                      font=("Microsoft YaHei", 13, "bold"))
        self.progress_pct.pack(side="right", padx=(0, 4))

        self.main_progress = ttk.Progressbar(progress_inner, orient="horizontal",
                                              mode="determinate", maximum=100)
        self.main_progress.pack(side="right", fill="x", expand=True, padx=(0, 10))
        self.main_progress["value"] = 0

        # Hide progress panel initially (show only during processing)
        self.progress_panel.pack_forget()
```

- [ ] **Step 7: Update `_set_step_state` to work with new indicator style**

Replace lines 181-194 (`_set_step_state` method):

```python
    def _set_step_state(self, step_id: str, state: str, desc: str = ""):
        self.step_state[step_id] = state
        outer = self.step_frames[step_id]
        num_label = outer.num_label
        if state == "done":
            num_label.config(bg=SUCCESS_COLOR, text="✓")
            outer.config(bg=SUCCESS_COLOR)
        elif state == "active":
            num_label.config(bg=ACCENT_COLOR, text=step_id)
            outer.config(bg=ACCENT_COLOR)
        else:
            num_label.config(bg=BORDER_COLOR, text=step_id)
            outer.config(bg=BORDER_COLOR)
        if desc:
            self.step_desc[step_id].set(desc)
        self._update_main_progress()
```

- [ ] **Step 8: Add `_update_main_progress` method and `_show_progress_panel` helper**

Add new methods after the existing `_set_step_state`:

```python
    def _update_main_progress(self):
        """Sync the unified progress bar with current step states."""
        active_steps = [sid for sid, st in self.step_state.items() if st == "active" or st == "done"]
        if not active_steps:
            self.progress_panel.pack_forget()
            return

        step_names = {"1": "视频分解与采样", "2": "入侵检测", "3": "关键帧筛选", "4": "视频重构"}
        current_step = "1"
        for sid in ["4", "3", "2", "1"]:
            if self.step_state[sid] in ("active", "done"):
                current_step = sid
                break

        total_pct = 0
        step_weights = {"1": 25, "2": 25, "3": 25, "4": 25}
        for sid, weight in step_weights.items():
            if self.step_state[sid] == "done":
                total_pct += weight
            elif self.step_state[sid] == "active":
                total_pct += int(weight * self.step_progress[sid].get() / 100)
                break
        total_pct = min(100, max(0, total_pct))

        self.progress_panel.pack(fill="x", pady=(4, 0), ipady=8,
                                  before=self.master.children.get("!frame", None))
        # Actually just repack it simply:
        self.progress_panel.pack(fill="x", pady=(4, 0), before=self.log_text.master)

        if self.step_state[current_step] == "active":
            self.progress_step_badge.config(bg=ACCENT_COLOR, text=current_step)
        elif self.step_state[current_step] == "done":
            self.progress_step_badge.config(bg=SUCCESS_COLOR, text="✓")

        self.progress_label.config(text=step_names.get(current_step, ""))
        self.progress_step_hint.config(text=f"步骤 {current_step}/4")
        self.main_progress["value"] = total_pct
        self.progress_pct.config(text=f"{total_pct}%")
```

Hmm, this approach is getting complex with Tkinter geometry management. Let me simplify: the progress panel should always be packed but toggle visibility based on whether processing is happening. Let me rethink.

Actually, let me keep it simpler. The progress panel will always be present in the layout — just show "就绪" when idle and show step info when processing. This avoids Tkinter pack/unpack complexity.

Let me rewrite the approach:

- [ ] **Step 6 (revised): Add bottom unified progress bar**

After the `log_text` pack (after line 141 in original), add:

```python

        # ── Unified progress bar ──
        progress_frame = tk.Frame(right_panel, bg=PANEL_COLOR, bd=1, relief="solid",
                                   highlightbackground=BORDER_COLOR)
        progress_frame.pack(fill="x", pady=(4, 0), ipady=8)

        progress_inner = tk.Frame(progress_frame, bg=PANEL_COLOR)
        progress_inner.pack(fill="x", padx=14, pady=8)

        self.progress_step_badge = tk.Label(progress_inner, text="—", bg=BORDER_COLOR, fg="white",
                                             font=("Microsoft YaHei", 10, "bold"), width=3, height=1)
        self.progress_step_badge.pack(side="left", padx=(0, 10))

        self.progress_label = tk.Label(progress_inner, text="就绪", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                        font=("Microsoft YaHei", 10, "bold"))
        self.progress_label.pack(side="left")

        self.progress_step_hint = tk.Label(progress_inner, text="", bg=PANEL_COLOR, fg=MUTED_COLOR,
                                            font=("Microsoft YaHei", 9))
        self.progress_step_hint.pack(side="left", padx=(8, 0))

        self.progress_pct = tk.Label(progress_inner, text="0%", bg=PANEL_COLOR, fg=ACCENT_COLOR,
                                      font=("Microsoft YaHei", 14, "bold"))
        self.progress_pct.pack(side="right", padx=(0, 4))

        self.main_progress = ttk.Progressbar(progress_inner, orient="horizontal",
                                              length=300, mode="determinate", maximum=100)
        self.main_progress.pack(side="right", fill="x", expand=True, padx=(0, 12))
```

- [ ] **Step 7 (revised): Update `_set_step_state`**

```python
    def _set_step_state(self, step_id: str, state: str, desc: str = ""):
        self.step_state[step_id] = state
        outer = self.step_frames[step_id]
        num_label = outer.num_label
        if state == "done":
            num_label.config(bg=SUCCESS_COLOR, fg="white", text="✓")
            outer.config(bg=SUCCESS_COLOR)
        elif state == "active":
            num_label.config(bg=ACCENT_COLOR, fg="white", text=step_id)
            outer.config(bg=ACCENT_COLOR)
        else:
            num_label.config(bg=BORDER_COLOR, fg="white", text=step_id)
            outer.config(bg=BORDER_COLOR)
        if desc:
            self.step_desc[step_id].set(desc)
        self._sync_unified_progress()
```

- [ ] **Step 8 (revised): Add `_sync_unified_progress` method**

```python
    def _sync_unified_progress(self):
        """Update the bottom unified progress bar from step states."""
        step_names = {"1": "视频分解与采样", "2": "入侵检测", "3": "关键帧筛选", "4": "视频重构"}
        # Each step contributes 25% of the total progress bar
        step_base = {"1": 0, "2": 25, "3": 50, "4": 75}

        # Find current active step (iterate 4→1 to find the rightmost active/done)
        for sid in ["4", "3", "2", "1"]:
            st = self.step_state[sid]
            if st == "done":
                if sid == "4":
                    # All steps complete
                    self.progress_step_badge.config(bg=SUCCESS_COLOR, fg="white", text="✓")
                    self.progress_label.config(text="处理完成")
                    self.progress_step_hint.config(text="")
                    self.main_progress["value"] = 100
                    self.progress_pct.config(text="100%")
                    return
                continue  # keep walking backward to find the active step
            elif st == "active":
                step_pct = self.step_progress[sid].get()
                overall = step_base[sid] + int(25 * step_pct / 100)
                self.progress_step_badge.config(bg=ACCENT_COLOR, fg="white", text=sid)
                self.progress_label.config(text=step_names[sid])
                self.progress_step_hint.config(text=f"步骤 {sid}/4")
                self.main_progress["value"] = overall
                self.progress_pct.config(text=f"{overall}%")
                return

        # Idle state (no step is active or done)
        self.progress_step_badge.config(bg=BORDER_COLOR, fg="white", text="—")
        self.progress_label.config(text="就绪")
        self.progress_step_hint.config(text="")
        self.main_progress["value"] = 0
        self.progress_pct.config(text="0%")
```

- [ ] **Step 9: Add `_open_camera` method and update `_start_processing` for auto-export**

Add `_open_camera` method (insert after `_select_video` method, around line 221):

```python
    def _open_camera(self):
        """Open the camera recording dialog."""
        cameras = CameraRecorder.list_cameras()
        if not cameras:
            messagebox.showwarning("警告", "未检测到可用摄像头")
            return
        camera_index = cameras[0]["index"]
        if len(cameras) > 1:
            # If multiple cameras, use the first — user can be prompted in future
            logging.info("检测到 %d 个摄像头，使用 Camera %d", len(cameras), camera_index)

        save_dir = self.app_config.camera_save_path
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        def on_ready(video_path: str):
            self.video_path.set(video_path)
            logging.info("摄像头视频已加载: %s", video_path)
            try:
                self.current_metadata = get_video_metadata(video_path)
                self.stat_output.set(Path(self.app_config.output_path).name)
                self.status_extra.set(f"输出: {self.app_config.output_path}")
                self.start_btn.config(state="normal")
                self._update_preview_hint()
            except Exception as e:
                logging.error("读取摄像头视频失败: %s", e)
                messagebox.showerror("错误", f"无法读取录制的视频: {e}")

        CameraDialog(self, save_dir, on_ready, camera_index=camera_index)
```

Need to also import CameraRecorder for the list_cameras call:

Add to the camera_recorder import line:
```python
from src.camera_recorder import CameraRecorder, CameraDialog
```

Now update `_start_processing` to respect `auto_export_video`. Find lines 285-286 in the original where `self._run_export()` is called:

```python
                # Step 4: Video export (if auto-export enabled)
                if self.app_config.auto_export_video:
                    self._run_export()
                else:
                    self.ui_queue.put(("step", {"id": "4", "state": "done", "desc": "导出已跳过（可在设置中开启）"}))
                self.ui_queue.put(("stats", {
                    "events": str(len(events)),
                    "keyframes": str(len(result.recommended_keyframes)),
                    "time": format_seconds(_time_module.time() - pipeline_start),
                }))
                self.ui_queue.put(("done", {}))
```

Wait, the original code at line 296 has the `except Exception` block. Let me look at the specific block to replace.

Original lines 284-297:
```python
                # Step 4: Video export
                self._run_export()
                self.ui_queue.put(("stats", {
                    "events": str(len(events)),
                    "keyframes": str(len(result.recommended_keyframes)),
                    "time": format_seconds(_time_module.time() - pipeline_start),
                }))
                self.ui_queue.put(("done", {}))
            except Exception as e:
```

Replace with:
```python
                # Step 4: Video export (respect auto_export setting)
                if self.app_config.auto_export_video:
                    self._run_export()
                else:
                    self.ui_queue.put(("step", {"id": "4", "state": "done",
                                                "desc": "跳过（可在设置中开启自动导出）"}))
                    self.ui_queue.put(("progress", {"step": "4", "current": 100, "total": 100}))
                self.ui_queue.put(("stats", {
                    "events": str(len(events)),
                    "keyframes": str(len(result.recommended_keyframes)),
                    "time": format_seconds(_time_module.time() - pipeline_start),
                }))
                self.ui_queue.put(("done", {}))
            except Exception as e:
```

- [ ] **Step 10: Update `_add_stat` to card style**

Replace lines 173-179:

```python
    def _add_stat(self, parent, col, var, label):
        card = tk.Frame(parent, bg=PANEL_COLOR, bd=1, relief="solid",
                         highlightbackground=BORDER_COLOR, highlightthickness=0)
        card.grid(row=0, column=col, sticky="ew", padx=6, pady=10, ipady=4)
        tk.Label(card, textvariable=var, bg=PANEL_COLOR, fg=ACCENT_COLOR,
                 font=("Microsoft YaHei", 18, "bold")).pack()
        tk.Label(card, text=label, bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 8)).pack(pady=(0, 4))
```

- [ ] **Step 11: Handle `handle_ui_action` — wire progress updates to unified bar**

In `handle_ui_action`, after updating `step_progress` (line 407), add a call to `_sync_unified_progress`:

At the end of the `progress` branch (after line 412), add:
```python
            self._sync_unified_progress()
```

- [ ] **Step 12: Update `refresh_from_config`**

At the end of the method, add camera save path directory creation:

```python
    def refresh_from_config(self):
        """Called when returning from settings page."""
        self.stat_output.set(Path(self.app_config.output_path).name)
        self.status_extra.set(f"输出: {self.app_config.output_path}")
```

No change needed here — it already works.

- [ ] **Step 13: Verify GUI starts without errors**

Run:
```bash
cd d:/All_Python/Project/torch_env_3 && python -c "import tkinter as tk; from src.gui_app import VideoProcessorApp; print('GUI module imports OK')"
```
Expected: prints success message, no Tkinter errors.

- [ ] **Step 14: Update button hover colors in `_build_ui`**

The `select_btn` is removed (replaced by dropdown). The `settings_btn` already uses activebackground. The `start_btn` and `export_btn` use activebackground matching their bg. All good from Step 3.

- [ ] **Step 15: Commit**

```bash
cd d:/All_Python/Project/torch_env_3 && git add src/gui_main.py && git commit -m "feat: modern UI refactor — dropdown video source, simplified step indicators, unified progress bar"
```

---

### Task 4: Update settings page with new camera and export settings

**Files:**
- Modify: `src/gui_settings.py`

- [ ] **Step 1: Add camera_save_path variable and path row**

In `_build_ui`, after the temp_var path row (after line 49), add the camera save path row.

Currently lines 46-49:
```python
        self.output_var = tk.StringVar()
        self.temp_var = tk.StringVar()
        self._add_path_row("输出文件默认路径", self.output_var, "所有处理结果将保存到此目录")
        self._add_path_row("临时文件路径", self.temp_var, "视频分解时的帧缓存目录")
```

Replace with:
```python
        self.output_var = tk.StringVar()
        self.temp_var = tk.StringVar()
        self.camera_var = tk.StringVar()
        self._add_path_row("输出文件默认路径", self.output_var, "所有处理结果将保存到此目录")
        self._add_path_row("临时文件路径", self.temp_var, "视频分解时的帧缓存目录")
        self._add_path_row("🆕 摄像头录制保存路径", self.camera_var, "摄像头录制的视频文件自动保存到此目录")
```

- [ ] **Step 2: Add auto-export checkbox**

In the preferences section (around lines 69-79), after the existing pref_frame checkboxes, add:

Current lines 72-79:
```python
        tk.Checkbutton(pref_frame, text="处理完成后自动打开输出文件夹", variable=self.auto_open_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)
        tk.Checkbutton(pref_frame, text="启动时自动加载上次的输出路径", variable=self.remember_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)
```

Replace with:
```python
        self.auto_export_var = tk.BooleanVar()
        tk.Checkbutton(pref_frame, text="处理完成后自动导出视频到输出路径", variable=self.auto_export_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)
        tk.Checkbutton(pref_frame, text="处理完成后自动打开输出文件夹", variable=self.auto_open_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)
        tk.Checkbutton(pref_frame, text="启动时自动加载上次的输出路径", variable=self.remember_var,
                       bg=SURFACE_COLOR, font=("Microsoft YaHei", 10),
                       activebackground=SURFACE_COLOR).pack(anchor="w", pady=2)
```

- [ ] **Step 3: Update `_load_config` to load new fields**

After line 124 (after `self.remember_var.set(c.remember_last_output)`), add:

```python
        self.camera_var.set(c.camera_save_path)
        self.auto_export_var.set(c.auto_export_video)
```

- [ ] **Step 4: Update `_save` to save new fields**

After line 139 (after `self.app_config.remember_last_output = self.remember_var.get()`), add:

```python
            self.app_config.camera_save_path = self.camera_var.get()
            self.app_config.auto_export_video = self.auto_export_var.get()
```

- [ ] **Step 5: Commit**

```bash
cd d:/All_Python/Project/torch_env_3 && git add src/gui_settings.py && git commit -m "feat: add camera save path and auto-export settings to settings page"
```

---

### Task 5: Update gui_app.py — wire camera_save_path and auto_export

**Files:**
- Modify: `src/gui_app.py:30-40` (config initialization and paths)

- [ ] **Step 1: Create camera_save_path directory on startup**

In `VideoProcessorApp.__init__`, after line 33 (`Path(self.app_config.output_path).mkdir(...)`), add:

```python
        Path(self.app_config.camera_save_path).mkdir(parents=True, exist_ok=True)
```

So the block (lines 33-34) becomes:
```python
        Path(self.app_config.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.app_config.temp_path).mkdir(parents=True, exist_ok=True)
        Path(self.app_config.camera_save_path).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Verify app launches**

Run (visual check — close manually):
```bash
cd d:/All_Python/Project/torch_env_3 && python -m src.gui_app
```
Expected: window opens with new colors and layout. Close manually.

- [ ] **Step 3: Commit**

```bash
cd d:/All_Python/Project/torch_env_3 && git add src/gui_app.py && git commit -m "feat: create camera_save_path directory on app startup"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `docs/课程大作业报告.md`
- Modify: `docs/课程大作业报告.docx`

- [ ] **Step 1: Add camera recording section to report markdown**

In `docs/课程大作业报告.md`, after the "多格式视频自适应帧采样与分解" bullet (line 52), add a new bullet:

At line 52, insert after the line `1. **多格式视频自适应帧采样与分解**：...`:

```markdown
1. **双源视频输入**：支持从本地文件系统选择视频文件（MP4/AVI/MKV/MOV/FLV/WMV）或通过摄像头实时录制视频（自动保存为MP4格式），录制窗口提供实时预览、录制时长显示和即时回放功能；
```

Then renumber the existing items 1-4 to accommodate. Actually, the existing content has items 1-4, and the first item already covers "多格式视频自适应帧采样与分解". So I should add this as an additional point or merge with point 1. Let me insert it as a NEW point between 1 and 2, shifting numbers:

At the bullet list in section 1.3 (lines 52-55), replace the entire list:

Old:
```markdown
本系统完成的主要功能包括：
1. **多格式视频自适应帧采样与分解**：支持MP4、AVI、MKV等常见视频格式，按可配置采样步长将视频分解为JPEG帧序列；
2. **入侵事件检测与时间轴标记**：基于MOG2自适应高斯混合模型进行背景建模与运动前景提取，结合HOG描述子实现行人目标检测，通过时序连续性分析将逐帧检测结果聚合为结构化事件；
3. **关键帧智能筛选**：使用ResNet-18预训练模型提取帧的512维深度特征向量，采用KMeans聚类算法对视觉相似帧进行分组，在各簇内选取距质心最近的帧作为代表性关键帧，消除视觉冗余；
4. **视频合成与导出**：基于FFmpeg的trim+concat filter_complex实现多片段高效合成，支持音频保留和降级OpenCV备选方案，按结构化目录导出摘要视频和独立事件片段。
```

New:
```markdown
本系统完成的主要功能包括：
1. **双源视频输入**：支持从本地文件系统选择视频文件（MP4/AVI/MKV/MOV/FLV/WMV）或通过摄像头实时录制视频并自动保存为MP4格式；录制窗口提供实时画面预览、录制时长指示和一键停止录制功能；
2. **多格式视频自适应帧采样与分解**：支持MP4、AVI、MKV等常见视频格式，按可配置采样步长将视频分解为JPEG帧序列；
3. **入侵事件检测与时间轴标记**：基于MOG2自适应高斯混合模型进行背景建模与运动前景提取，结合HOG描述子实现行人目标检测，通过时序连续性分析将逐帧检测结果聚合为结构化事件；
4. **关键帧智能筛选**：使用ResNet-18预训练模型提取帧的512维深度特征向量，采用KMeans聚类算法对视觉相似帧进行分组，在各簇内选取距质心最近的帧作为代表性关键帧，消除视觉冗余；
5. **视频合成与导出**：基于FFmpeg的trim+concat filter_complex实现多片段高效合成，支持音频保留和降级OpenCV备选方案，按结构化目录导出摘要视频和独立事件片段。
```

- [ ] **Step 2: Update functional requirements in section 3.1**

In section 3.1 (line 173), insert a new requirement before item 1:

At line 174, add before `1. **视频输入与元数据提取**`:

```markdown
1. **双源视频采集**：支持本地视频文件选择（MP4/AVI/MKV/MOV/FLV/WMV）和摄像头实时录制两种视频来源。摄像头录制通过独立弹窗控制，录制内容自动保存为MP4格式至可配置路径，录制完成后自动加载至主界面预览区。
```

Then shift existing items 1-7 to 2-8.

- [ ] **Step 3: Update system architecture diagram (section 3.2)**

In the architecture diagram at line 195, add the camera module:

```
│   gui_app.py (应用框架)                                │
│   gui_main.py (主处理页)   gui_settings.py (设置页)     │
│   camera_recorder.py (摄像头录制)                       │
```

- [ ] **Step 4: Update module description in section 4.3**

After the core interface diagram (line 377), add:

```markdown
    ├── camera_recorder.CameraRecorder()
    │       .open() / .start_recording() / .stop_recording()
    │
    ├── camera_recorder.CameraDialog()
    │       弹出摄像头录制窗口，录制完成后回调主界面
```

- [ ] **Step 5: Add camera recording detail in section 4.5**

After section 4.5.4, add a new subsection:

```markdown
#### 4.5.5 摄像头录制（camera_recorder.py）

`CameraRecorder`类封装了OpenCV摄像头采集与MP4视频编码写入功能。核心设计如下：

1. **设备枚举**：`list_cameras()`方法检测系统中可用的摄像头设备，返回各设备的索引、分辨率和名称；
2. **实时预览**：`CameraDialog`继承`tk.Toplevel`，通过定时器（50ms间隔）循环读取摄像头帧并更新Tkinter Label显示，实现流畅的实时预览；
3. **录制控制**：提供「开始录制」「停止录制」「取消」三个按钮。录制时在预览画面右上角显示红色录制指示器和计时器；
4. **异步写入**：录制帧在预览刷新时同步写入VideoWriter，录制完成后通过回调函数将视频路径传递给主界面，自动加载并更新预览信息。

录制视频默认保存至项目根目录下的`saved_video/`目录，文件命名格式为`camera_YYYYMMDD_HHMMSS.mp4`，保存路径可在系统设置中自定义。
```

- [ ] **Step 6: Update conclusion section (section 7)**

In the conclusion (line 602), update the first sentence to mention camera recording:

Old:
```markdown
本文使用Python语言，基于PyTorch、OpenCV和scikit-learn等技术框架，设计并实现了一个面向监控视频领域的智能浓缩与异常筛查系统。系统主要功能包括：多格式视频的自适应帧采样与分解、基于MOG2+HOG的入侵事件检测与时间轴标记、基于ResNet-18深度特征与KMeans无监督聚类的关键帧去重与推荐，以及基于FFmpeg的多片段视频合成导出。
```

New:
```markdown
本文使用Python语言，基于PyTorch、OpenCV和scikit-learn等技术框架，设计并实现了一个面向监控视频领域的智能浓缩与异常筛查系统。系统主要功能包括：双源视频输入（本地文件与摄像头实时录制）、多格式视频的自适应帧采样与分解、基于MOG2+HOG的入侵事件检测与时间轴标记、基于ResNet-18深度特征与KMeans无监督聚类的关键帧去重与推荐，以及基于FFmpeg的多片段视频合成导出。
```

- [ ] **Step 7: Update the future improvements section**

In section 7, item 5 (line 613):
Old item 5 reads: `5. 支持视频流的实时处理，而非仅限于离线文件；`

Replace with:
```markdown
5. 增强摄像头录制功能，支持录制过程中实时进行入侵检测和关键帧筛选，而非录制完成后离线处理；
```

- [ ] **Step 8: Commit markdown report**

```bash
cd d:/All_Python/Project/torch_env_3 && git add docs/课程大作业报告.md && git commit -m "docs: add camera recording feature to course report"
```

- [ ] **Step 9: Update the .docx file**

The `.docx` file is a binary Word document. Use `python-docx` to update it based on the .md content.

Run:
```bash
cd d:/All_Python/Project/torch_env_3 && python -c "
from docx import Document
doc = Document('docs/课程大作业报告.docx')
# The docx should mirror the md file structure.
# Since the .docx is a templated report, we'll note that it needs manual
# or scripted sync. For now, log that the .md has been updated.
print('Note: .docx sync requires manual editing or a docx generation script.')
print('The .md file has been fully updated with camera recording content.')
print('Please regenerate .docx from .md using pandoc or manually update.')
"
```

If you have `pandoc` available, convert .md to .docx:
```bash
cd d:/All_Python/Project/torch_env_3 && pandoc docs/课程大作业报告.md -o docs/课程大作业报告.docx 2>/dev/null && echo "✅ .docx regenerated from .md" || echo "⚠ pandoc not available, please update .docx manually"
```

- [ ] **Step 10: Commit docx if regenerated**

```bash
cd d:/All_Python/Project/torch_env_3 && git add docs/课程大作业报告.docx 2>/dev/null; git commit -m "docs: sync .docx with camera recording updates" || echo "no docx changes to commit"
```

---

### Task 7: Final verification and integration test

**Files:**
- Test: manual launch and smoke test

- [ ] **Step 1: Launch the app and verify all changes**

Run:
```bash
cd d:/All_Python/Project/torch_env_3 && python -m src.gui_app
```

Manual checks:
1. ✅ New color theme (light gray bg, blue accents, white cards)
2. ✅ Top bar has `📁 视频来源 ▾` dropdown, `⚙ 设置`, `▶ 开始处理`, `📤 导出`
3. ✅ Left panel has compact step indicators (no progress bars)
4. ✅ Bottom of right panel has unified progress bar showing "就绪"
5. ✅ Click `📁 视频来源 ▾` → shows `📂 从文件选择` and `🎥 从摄像头录制`
6. ✅ Click settings → new `摄像头录制保存路径` field and `自动导出` checkbox
7. ✅ Button text is clearly readable (white on blue, dark on white)
8. ✅ Select a video file, click start processing → unified progress bar updates
9. ✅ Export works as expected

- [ ] **Step 2: Run full smoke test (file mode)**

```bash
cd d:/All_Python/Project/torch_env_3 && python -c "
from src.config import load_app_config
c = load_app_config()
print('Config OK')
print(f'  output_path: {c.output_path}')
print(f'  temp_path: {c.temp_path}')
print(f'  camera_save_path: {c.camera_save_path}')
print(f'  auto_export_video: {c.auto_export_video}')
from src.camera_recorder import CameraRecorder, CameraDialog
print('Camera module OK')
from src.gui_main import MainPage
print('MainPage OK')
from src.gui_settings import SettingsPage
print('SettingsPage OK')
print('All imports successful')
"
```
Expected: all OK.

- [ ] **Step 3: Commit final verification**

```bash
cd d:/All_Python/Project/torch_env_3 && git status && git log --oneline -5
```

---

## Summary of Changes

| # | File | Change |
|---|------|--------|
| 1 | `src/config.py` | Color theme (7 constants updated, 1 added); AppConfig +2 fields |
| 2 | `src/camera_recorder.py` | **NEW** — CameraRecorder class + CameraDialog Toplevel |
| 3 | `src/gui_main.py` | Major: dropdown menu, simplified left panel, unified progress bar, _open_camera, auto_export toggle |
| 4 | `src/gui_settings.py` | +camera save path row, +auto_export checkbox |
| 5 | `src/gui_app.py` | +camera_save_path dir creation |
| 6 | `docs/课程大作业报告.md` | +camera recording sections throughout |
| 7 | `docs/课程大作业报告.docx` | Sync from .md |
