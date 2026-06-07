"""Application framework — window, page switching, queue polling."""
import logging
import queue
import tkinter as tk
from pathlib import Path

from src.config import (AppConfig, ProcessingConfig, SURFACE_COLOR,
                        LOG_DIR, PROGRESS_POLL_MS, load_app_config)
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
        self.app_config = load_app_config()
        Path(self.app_config.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.app_config.temp_path).mkdir(parents=True, exist_ok=True)

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
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    app = VideoProcessorApp()
    app.run()
