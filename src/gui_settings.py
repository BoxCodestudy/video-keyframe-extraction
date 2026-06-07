"""Settings page — output paths, processing parameters, UI preferences."""
import tkinter as tk
from tkinter import ttk, filedialog
from src.config import (AppConfig, save_app_config, ACCENT_COLOR, SURFACE_COLOR,
                        PANEL_COLOR, TEXT_COLOR, MUTED_COLOR, BORDER_COLOR)


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
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Button(topbar, text="← 返回", bg=PANEL_COLOR, fg=TEXT_COLOR,
                  font=("Microsoft YaHei", 10), bd=0, cursor="hand2",
                  activebackground=PANEL_COLOR, command=self.on_back).pack(side="left", padx=16, pady=10)
        tk.Label(topbar, text="⚙ 系统设置", bg=PANEL_COLOR, fg=TEXT_COLOR,
                 font=("Microsoft YaHei", 14, "bold")).pack(side="left", padx=4)
        tk.Button(topbar, text="保存设置", bg=ACCENT_COLOR, fg="white",
                  font=("Microsoft YaHei", 10, "bold"), bd=0, padx=16, pady=6,
                  cursor="hand2", activebackground="#1557b0", command=self._save).pack(side="right", padx=16, pady=10)

        # ── Scrollable content ──
        canvas = tk.Canvas(self, bg=SURFACE_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.content = tk.Frame(canvas, bg=SURFACE_COLOR)
        self.content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>",
            lambda ev: canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # ── Section: Paths ──
        self._add_section("文件路径")
        self.output_var = tk.StringVar()
        self.temp_var = tk.StringVar()
        self._add_path_row("输出文件默认路径", self.output_var, "所有处理结果将保存到此目录")
        self._add_path_row("临时文件路径", self.temp_var, "视频分解时的帧缓存目录")

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
        tk.Frame(self.content, bg=BORDER_COLOR, height=1).pack(fill="x", padx=24, pady=(16, 8))
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
        tk.Button(entry_row, text="浏览...", bg=PANEL_COLOR, fg=ACCENT_COLOR,
                  font=("Microsoft YaHei", 9), bd=1, padx=10, cursor="hand2",
                  command=lambda v=var: self._browse_dir(v)).pack(side="left", padx=(6, 0))
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
