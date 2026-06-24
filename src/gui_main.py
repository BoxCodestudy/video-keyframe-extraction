"""Main processing page — video pipeline UI."""
import logging
import queue
import threading
import time as _time_module
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from PIL import Image, ImageTk

from src.config import (AppConfig, ProcessingConfig, AnalysisResult, VideoMetadata,
                        IntrusionEvent, SURFACE_COLOR, PANEL_COLOR, TEXT_COLOR,
                        MUTED_COLOR, ACCENT_COLOR, SUCCESS_COLOR, WARNING_COLOR,
                        BORDER_COLOR, PROGRESS_POLL_MS)
from src.detection import detect_intrusion_events, map_events_to_image_files
from src.features import CNNFeatureExtractor
from src.keyframes import auto_select_keyframes_by_clustering
from src.video_io import get_video_metadata, video_to_images
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

        self.step_state = {"1": "waiting", "2": "waiting", "3": "waiting", "4": "waiting"}
        self.step_desc = {str(i): tk.StringVar(value="等待开始") for i in range(1, 5)}
        self.step_progress = {str(i): tk.DoubleVar(value=0) for i in range(1, 5)}

        self.stat_events = tk.StringVar(value="--")
        self.stat_keyframes = tk.StringVar(value="--")
        self.stat_time = tk.StringVar(value="--")
        self.stat_output = tk.StringVar(value="--")

        self.log_expanded = tk.BooleanVar(value=True)
        self._preview_index = 0

        self._build_ui()
        self._poll_log()

    def _build_ui(self):
        # ── Top Bar ──
        topbar = tk.Frame(self, bg=PANEL_COLOR)
        topbar.pack(fill="x")
        tk.Label(topbar, text="◈ 基于深度特征聚类的视频关键帧提取与事件检测系统", bg=PANEL_COLOR, fg=ACCENT_COLOR,
                 font=("Microsoft YaHei", 13, "bold")).pack(side="left", padx=16, pady=10)
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
                                   padx=14, pady=4, cursor="hand2", command=self._start_processing,
                                   state="disabled")
        self.start_btn.pack(side="left", padx=4)

        # ── Main Content ──
        main = tk.Frame(self, bg=SURFACE_COLOR)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: Steps
        left_panel = tk.Frame(main, bg=PANEL_COLOR, width=260)
        left_panel.pack(side="left", fill="y", padx=(0, 8))
        left_panel.pack_propagate(False)
        tk.Label(left_panel, text="处理流程", bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 8))
        step_names = ["视频分解与采样", "入侵检测", "关键帧筛选", "视频重构"]
        self.step_frames = {}
        for i, name in enumerate(step_names, 1):
            self.step_frames[str(i)] = self._create_step_card(left_panel, str(i), name)

        # Right: Preview + Stats + Log
        right_panel = tk.Frame(main, bg=SURFACE_COLOR)
        right_panel.pack(side="left", fill="both", expand=True)

        preview_frame = tk.Frame(right_panel, bg=PANEL_COLOR, bd=0)
        preview_frame.pack(fill="both", expand=True, pady=(0, 4))
        self.preview_label = tk.Label(preview_frame, bg=PANEL_COLOR, anchor="center",
                                       text="🎬\n拖拽视频文件到此处 或 点击\"选择视频\"开始\n支持 MP4 / AVI / MKV / MOV",
                                       fg=MUTED_COLOR, font=("Microsoft YaHei", 12))
        self.preview_label.pack(fill="both", expand=True, padx=20, pady=20)
        self.preview_label.bind("<MouseWheel>", self._on_preview_scroll)
        self.preview_label.bind("<Button-4>", self._on_preview_scroll)
        self.preview_label.bind("<Button-5>", self._on_preview_scroll)

        stats_frame = tk.Frame(right_panel, bg=PANEL_COLOR)
        stats_frame.pack(fill="x", pady=(0, 4))
        for i in range(4):
            stats_frame.grid_columnconfigure(i, weight=1)
        self._add_stat(stats_frame, 0, self.stat_events, "检测事件")
        self._add_stat(stats_frame, 1, self.stat_keyframes, "关键帧")
        self._add_stat(stats_frame, 2, self.stat_time, "处理耗时")
        self._add_stat(stats_frame, 3, self.stat_output, "输出路径")

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
        self.log_text.pack(fill="x")

        # Bottom status bar
        bottombar = tk.Frame(self, bg=PANEL_COLOR, height=28, bd=0)
        bottombar.pack(fill="x", side="bottom")
        bottombar.pack_propagate(False)
        tk.Label(bottombar, text="●", bg=PANEL_COLOR, fg=SUCCESS_COLOR,
                 font=("", 10)).pack(side="left", padx=(12, 4))
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
        tk.Label(card, textvariable=self.step_desc[step_id], bg=PANEL_COLOR, fg=MUTED_COLOR,
                 font=("Microsoft YaHei", 8), anchor="w").pack(fill="x", padx=10, pady=(0, 4))
        progress = ttk.Progressbar(card, variable=self.step_progress[step_id], maximum=100)
        progress.pack(fill="x", padx=10, pady=(0, 8))
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
                text=f"已加载: {Path(m.path).name}\n{m.width}x{m.height} | {m.fps:.1f} fps | {m.frame_count} 帧\n点击\"▶ 开始处理\"启动分析",
                fg=TEXT_COLOR)

    def _start_processing(self):
        if not self.video_path.get():
            messagebox.showwarning("警告", "请先选择视频文件")
            return
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            messagebox.showwarning("警告", "请等待当前处理完成")
            return
        for sid in ["1", "2", "3", "4"]:
            self._set_step_state(sid, "waiting", "等待开始")
            self.step_progress[sid].set(0)
        self.stat_events.set("--")
        self.stat_keyframes.set("--")
        self.stat_time.set("--")
        self.start_btn.config(state="disabled")
        self._set_step_state("1", "active", "分解中...")
        pipeline_start = _time_module.time()

        def worker():
            try:
                self.ui_queue.put(("step", {"id": "1", "state": "active", "desc": "分解中..."}))
                temp_dir = self.app_config.temp_path or "temp/frames"
                metadata = video_to_images(
                    self.video_path.get(), temp_dir, self.proc_config.sample_stride,
                    progress_callback=lambda **kw: self.ui_queue.put(("progress", {"step": "1", **kw})),
                    max_width=self.proc_config.max_width, max_height=self.proc_config.max_height,
                )
                self.current_metadata = metadata
                self.source_sampled_frames = list_image_files(Path(temp_dir))
                self.ui_queue.put(("step", {"id": "1", "state": "done", "desc": f"提取 {len(self.source_sampled_frames)} 帧"}))

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

                self.ui_queue.put(("step", {"id": "4", "state": "done", "desc": "处理完成，可导出"}))
                self.ui_queue.put(("stats", {
                    "events": str(len(events)),
                    "keyframes": str(len(result.recommended_keyframes)),
                    "time": format_seconds(_time_module.time() - pipeline_start),
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
        self._preview_index = max(0, min(len(self.current_selected_frames) - 1, self._preview_index + delta))
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
