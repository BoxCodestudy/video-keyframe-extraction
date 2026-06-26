"""Camera recording — OpenCV capture with Tkinter recording dialog."""
import logging
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import config colors; fallback if run standalone
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

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the current capture resolution as (width, height)."""
        return (self._frame_width, self._frame_height)

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

    def get_frame(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None if capture fails."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            logging.warning("帧颜色转换失败 (cvtColor)")
            return None

    def start_recording(self, save_dir: str) -> str:
        """Begin writing video to save_dir. Returns the output file path."""
        try:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            logging.warning("无法创建录制目录: %s", save_dir)
            return ""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{stamp}.mp4"
        self._output_path = str(Path(save_dir) / filename)

        # Release any previous writer before creating a new one
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        self._writer = cv2.VideoWriter(
            self._output_path, fourcc, self.fps,
            (self._frame_width, self._frame_height),
        )
        if not self._writer.isOpened():
            logging.error("无法打开视频写入器: %s", self._output_path)
            self._writer.release()
            self._writer = None
            self._recording = False
            return ""
        self._recording = True
        logging.info("开始录制: %s", self._output_path)
        return self._output_path

    def write_frame(self, frame) -> bool:
        """Write a frame to the recording. Frame should be RGB numpy array."""
        if not self._recording or self._writer is None:
            return False
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return self._writer.write(bgr)

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
        self._recording_timer_id: Optional[str] = None
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

        # Recording indicator (overlaid — just a label for Tkinter)
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
        w, h = self._recorder.resolution
        info_frame = tk.Frame(self, bg="#0f172a")
        info_frame.pack(fill="x", padx=16, pady=(4, 12))
        self._info_label = tk.Label(info_frame, text=f"📐 {w}×{h}  ⚡ {RECORDING_FPS:.0f} fps  💾 {self.save_dir}  📝 MP4",
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
        self._recording_timer_id = self.after(1000, self._update_recording_timer)

    def _cleanup(self, call_callback: bool = True):
        """Common cleanup: stop recorder, cancel timers, destroy dialog."""
        if self._recorder.is_recording():
            self._recorder.stop_recording()
        self._recorder.release()
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id = None
        if self._recording_timer_id:
            self.after_cancel(self._recording_timer_id)
            self._recording_timer_id = None
        output_path = self._recorder._output_path
        self.destroy()
        if call_callback and output_path and Path(output_path).exists():
            logging.info("录制完成，视频保存至: %s", output_path)
            self.on_video_ready(output_path)

    def _stop_and_return(self):
        logging.info("用户停止录制")
        self._cleanup(call_callback=True)

    def _on_cancel(self):
        logging.info("摄像头录制已取消")
        self._cleanup(call_callback=False)
