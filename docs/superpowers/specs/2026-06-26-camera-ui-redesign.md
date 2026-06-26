# 摄像头录制 + UI 现代化改造 设计文档

**日期:** 2026-06-26
**状态:** 已批准

---

## 1. 目标

在现有视频关键帧提取与事件检测系统中增加以下能力：
- 实时摄像头录制作为视频来源
- UI 现代化改造（现代简洁风 + 进度条重构）
- 自动导出与路径配置增强
- 同步更新课程报告文档

## 2. 新增模块：摄像头录制 (`src/camera_recorder.py`)

### 2.1 职责
封装 OpenCV 摄像头捕获与 MP4 视频写入，提供可复用的录制窗口。

### 2.2 核心类

```python
class CameraRecorder:
    """管理摄像头采集和视频编码写入"""
    def __init__(self, save_dir: str, fps: float = 30.0, codec: str = "mp4v"):
        ...
    def list_cameras() -> list[dict]:           # 枚举可用摄像头
    def open(self, camera_index: int = 0) -> bool: ...
    def start_recording(self, save_dir: str) -> str: ...  # 返回保存路径
    def stop_recording(self) -> str: ...                   # 返回保存路径
    def get_frame(self) -> np.ndarray | None: ...
    def release(self): ...

class CameraDialog(tk.Toplevel):
    """摄像头录制弹窗 — 实时预览 + 录制控制"""
    def __init__(self, parent, save_dir: str, on_video_ready: callable): ...
    # 包含：预览画布、开始/停止/取消按钮、录制时长/状态显示
```

### 2.3 录制流程

1. 点击 `📁 视频来源 ▾ → 🎥 从摄像头录制` → 创建 `CameraDialog`
2. 对话框打开摄像头，显示实时预览（定时刷新 Canvas 或 Label）
3. 用户点击 `🔴 开始录制` → 开始写入 MP4 到 `saved_video/` 目录
4. 录制指示器（红点 + 时长）显示在预览画面上方
5. 用户点击 `⏹ 停止录制` → 停止写入，关闭摄像头流
6. 调用 `on_video_ready(video_path)` 回调 → 传递给 MainPage 加载
7. `✕ 取消` → 关闭摄像头，不保存

### 2.4 保存路径

默认 `{项目根目录}/saved_video/`，文件名 `camera_YYYYMMDD_HHMMSS.mp4`。
路径可在设置页面中修改。

## 3. UI 现代化改造 (`src/gui_main.py`, `src/gui_settings.py`, `src/gui_app.py`, `src/config.py`)

### 3.1 色彩主题（现代简洁风）

```
背景:        #f5f7fa  (浅灰蓝，替代 #f0f2f5)
卡片/面板:   #ffffff  (白色，替代原白)
主色调:      #2563eb  (蓝-600，替代 #1a73e8)
主色调hover: #1d4ed8  (蓝-700)
成功:        #16a34a  (绿-600，替代 #0ea854)
警告:        #d97706  (琥珀-600，替代 #e37400)
文字主色:    #1e293b  (深石板色，替代 #333333)
次要文字:    #64748b  (石板-500，替代 #888888)
边框:        #e2e8f0  (石板-200，替代 #e0e4e8)
面板色:      #ffffff  (保持)
```

按钮样式规则：
- **主按钮（开始处理、导出）**: `bg=#2563eb fg=white font=bold bd=0 padx=16 pady=6` — 白字蓝底
- **次要按钮（设置、视频来源）**: `bg=white fg=#475569 bd=1 relief=solid` — 深色字白底灰边框
- **危险按钮（开始录制）**: `bg=#ef4444 fg=white` — 红底白字

### 3.2 顶部工具栏改造

```
[◈ 系统标题]                    [📁 视频来源 ▾] [⚙ 设置] [▶ 开始处理] [📤 导出]
```

- `📁 视频来源 ▾`：Menubutton 或 Button + Menu 下拉 → `📂 从文件选择` / `🎥 从摄像头录制`
- 移除原来独立的两个按钮，合并为下拉菜单

### 3.3 左侧面板改造（宽度 260 → 180px）

原来：4 个步骤卡片，每个卡片带标题 + 描述文字 + 独立进度条
改为：4 个步骤指示器，只带圆形序号 + 步骤名 + 状态文字 + 左边框颜色指示

```
处理步骤

┃ ✓ 视频分解与采样
┃   提取 1800 帧

┃ 2 入侵检测          ← 蓝色左边框 = 进行中
┃   检测中...

┃ 3 关键帧筛选        ← 灰色左边框 = 等待中

┃ 4 视频重构          ← 灰色左边框 = 等待中
```

- 圆形序号：等待中=`#cbd5e1` 激活中=`#2563eb` 完成=`#16a34a`(显示✓)
- 左边框颜色与序号同步
- 去掉 `ttk.Progressbar`

### 3.4 底部统一进度条（新增）

在右侧面板下方、状态栏上方，新增一条宽进度条面板：

```
┌──────────────────────────────────────────────────────────────┐
│ (2) 入侵检测      步骤 2/4    ████████████░░░░░░  67%         │
└──────────────────────────────────────────────────────────────┘
```

- 白色卡片背景 + 圆角边框
- 左侧：当前步骤图标 + 名称 + "步骤 2/4"
- 中间：`ttk.Progressbar` 或自定义 Canvas 进度条
- 右侧：百分比数字
- 处理不在进行时隐藏或显示"就绪"

### 3.5 统计区改善

保持 4 格统计，改为卡片式：
- 白色卡片 + 微妙阴影 / 边框
- 数值用蓝色大字，标签用灰色小字

### 3.6 设置页面改造

新增项：
- `摄像头录制保存路径` — 带浏览按钮的路径选择行（默认 `{APP_DIR}/saved_video`）
- `☑ 处理完成后自动导出视频到输出路径` — 复选框（默认勾选）

保留项不变。

## 4. 自动导出逻辑

### 4.1 配置扩展 (`src/config.py`)

`AppConfig` 新增字段：
```python
camera_save_path: str = ""           # 默认 __post_init__ 中设为 saved_video/
auto_export_video: bool = True       # 处理完成后自动导出
```

`ProcessingConfig` 无需变动。

### 4.2 流水线行为

当前：步骤 4（视频导出）总是作为流水线的一部分在 `_start_processing` 中调用（`_run_export()`），所以导出已经在处理完成时发生了。但 export_btn 在流水线处理过程中被禁用。

改进：
- 流水线的步骤 4 照常执行（因为 _run_export 已经在 worker 中被调用）
- `auto_export_video` 控制的是：如果为 False，步骤 4 跳过（不作为流水线的一部分）
- 无论如何，导出按钮在处理完成后都处于可用状态

实际上看现有代码，`_start_processing` 中的 `worker()` 函数在第 285 行调用了 `self._run_export()`，所以步骤 4 已经是流水线的一部分。我们只需要：
1. 如果 `auto_export_video=False`，跳过 `_run_export()` 调用
2. 用 `auto_export_video` 配置控制这个行为

### 4.3 导出路径

不变：始终使用 `app_config.output_path` 作为导出根目录。用户可在设置中修改。

## 5. 交互流程总结

### 文件模式
```
📁 视频来源 ▾ → 📂 从文件选择 → 文件对话框 → 预览加载 → ▶ 开始处理 → 4步流水线 → 自动导出 → ✓ 完成
```

### 摄像头模式
```
📁 视频来源 ▾ → 🎥 从摄像头录制 → 录制窗口打开 → 🔴 开始录制 → ⏹ 停止录制
→ 窗口关闭 → 视频自动加载 → ▶ 开始处理 → 4步流水线 → 自动导出 → ✓ 完成
```

## 6. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/camera_recorder.py` | **新增** | 摄像头录制模块 |
| `src/config.py` | 修改 | 颜色常量 + camera_save_path + auto_export_video |
| `src/gui_main.py` | 修改 | UI 布局重构 + 视频来源下拉 + 底部进度条 + 摄像头集成 |
| `src/gui_settings.py` | 修改 | 新增 camera_save_path 和 auto_export 配置项 |
| `src/gui_app.py` | 修改 | 颜色 + 新增配置传递 |
| `docs/课程大作业报告.md` | 修改 | 新增摄像头录制相关内容 |
| `docs/课程大作业报告.docx` | 修改 | 基于 .md 同步更新 |

## 7. 不变的部分

- 所有后端算法模块（`detection.py`, `features.py`, `keyframes.py`, `video_io.py`, `utils.py`）
- `src/config.py` 的核心数据结构
- 设置页面的整体结构
- 多线程流水线架构
- logging 系统

## 8. 风险与注意事项

- Tkinter 不支持 CSS 渐变和复杂阴影，设计实现需降级到纯色 + 边框
- 摄像头录制依赖 OpenCV，需在启动时检测摄像头可用性
- Windows 上摄像头索引通常为 0，多摄像头场景提供选择
- 中文 Windows 系统字体：使用 `Microsoft YaHei` 作为默认
