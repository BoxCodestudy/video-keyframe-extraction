# 代码结构与 UI 优化设计文档

**日期**: 2026-06-07
**范围**: 单文件模块化拆分、性能优化、UI 重新设计

---

## 一、代码结构优化 — 模块化拆分

当前 `src/main.py`（1422行）拆分为以下文件：

| 文件                    | 职责     | 包含内容                                                                                  |
| --------------------- | ------ | ------------------------------------------------------------------------------------- |
| `src/config.py`       | 配置管理   | `AppConfig`, 默认参数常量, 配置加载/保存, `VideoMetadata`, `AnalysisResult`, `IntrusionEvent` 数据类 |
| `src/detection.py`    | 入侵检测   | `detect_intrusion_events()`, MOG2/HOG 调用, 事件聚合逻辑                                      |
| `src/features.py`     | 特征提取   | `CNNFeatureExtractor` 类, ResNet-18 加载, 特征提取, 批量处理                                     |
| `src/keyframes.py`    | 关键帧选择  | `select_frames_by_anchor()`, KMeans 聚类, 均匀采样回退                                        |
| `src/video_io.py`     | 视频 I/O | `video_to_images()`, `build_segments_*()`, 视频解码/编码, FFmpeg 调用                         |
| `src/gui_main.py`     | 主处理页面  | `MainPage` 类, 视频选择/拖拽, 四步骤流程, 进度条, 统计面板, 日志                                           |
| `src/gui_settings.py` | 设置页面   | `SettingsPage` 类, 路径配置, 参数设置, 界面偏好                                                    |
| `src/gui_app.py`      | 应用框架   | `VideoProcessorApp` 类, 窗口管理, 页面切换, 队列轮询                                               |
| `src/utils.py`        | 工具函数   | `sanitize_name()`, `format_event_time()`, `evenly_spaced_keyframes()`, 日志初始化          |

### 依赖关系

```
config.py          ← 无内部依赖
utils.py           ← 无内部依赖
detection.py       ← config.py
features.py        ← config.py
keyframes.py       ← config.py, features.py
video_io.py        ← config.py, detection.py
gui_settings.py    ← config.py
gui_main.py        ← config.py, detection.py, features.py, keyframes.py, video_io.py
gui_app.py         ← gui_main.py, gui_settings.py, config.py
```

---

## 二、性能优化

1. **ThreadPoolExecutor 复用**: `features.py` 中提取循环创建单例线程池，不用每批次新建
2. **视频跳帧读取**: `video_io.py` 使用 `cv2.CAP_PROP_POS_FRAMES` 跳帧，减少无效解码
3. **低运动帧跳过**: `detection.py` 预检测运动量，对完全静止帧跳过 HOG 人物检测
4. **删除死代码**: 移除 `preload_radius`, `preview_cache`, `preview_cache_order`, `processing_lock`, `preview_hover_active`（从未使用）
5. **ui_queue 消费**: 修复 `gui_app.py` 中 `poll_queues()` 添加 ui_queue 消费逻辑

---

## 三、UI 设计 — 现代扁平浅色

### 整体风格

- 现代扁平设计，浅色主题
- 主色 `#1a73e8`（蓝色），状态色：绿 `#0ea854`，黄 `#e37400`
- 圆角卡片、清晰分区、扁平图标
- 字体：Segoe UI / Microsoft YaHei 自动适配

### 页面结构（两页，同一窗口 Frame 切换）

**主处理页面**：

```
┌────────────────────────────────────────────────┐
│ ◈ Logo/标题         ⚙ 设置  📂 选择视频  ▶ 开始处理 │  顶栏
├──────────┬─────────────────────────────────────┤
│ 步骤1  ✓ │                                     │
│ 步骤2  ● │       预览区（空状态有引导提示）        │  主内容区
│ 步骤3  ○ │                                     │
│ 步骤4  ○ │                                     │
│          ├─────────────────────────────────────┤
│ 参数设置  │  检测事件:3  关键帧:12  耗时:1:23      │  统计栏
│          ├─────────────────────────────────────┤
│          │  📋 运行日志（可折叠）                  │  日志区
├──────────┴─────────────────────────────────────┤
│ ● 就绪              输出: D:/VideoOutput/  │ CPU   │  状态栏
└────────────────────────────────────────────────┘
```

**设置页面**：

```
┌────────────────────────────────────────────────┐
│ ← 返回           ⚙ 系统设置             保存设置  │  顶栏
├────────────────────────────────────────────────┤
│  输出文件默认路径     [________________] [浏览]   │
│  临时文件路径         [________________] [浏览]   │
│  ─────────────────────────────────────────────  │
│  默认处理参数                                    │
│    采样帧间隔 [10]    入侵阈值 [0.45]              │
│    最大分辨率 [960]   最小事件时长 [0.5]            │
│    运动权重 [0.65]    人物权重 [0.35]              │
│  ─────────────────────────────────────────────  │
│  界面偏好                                       │
│    ☑ 启动时加载上次输出路径                        │
│    ☑ 处理完成后自动打开输出文件夹                   │
└────────────────────────────────────────────────┘
```

### 步骤卡片状态

- **等待**: 灰色，无进度条
- **进行中**: 蓝色左边框 + 蓝色圆形编号 + 嵌入进度条
- **完成**: 绿色编号 ✓ + 整体淡化

### 按钮状态

- **主操作**（开始处理）: 实心蓝色背景
- **次要操作**（选择视频、设置）: 白色描边
- **引导操作**（选择视频未选时）: 橙色描边突出
- **禁用**: 灰色背景，无文字颜色

---

## 四、配置文件

设置保存在 `<output_path>/.video_processor_config.json`，每次启动自动加载。格式：

```json
{
  "output_path": "D:/VideoOutput/",
  "temp_path": "./temp/",
  "sample_stride": 10,
  "intrusion_threshold": 0.45,
  "max_width": 960,
  "min_event_duration": 0.5,
  "motion_weight": 0.65,
  "person_weight": 0.35,
  "auto_open_output": true,
  "remember_last_output": true
}
```

---

## 五、不涉及的内容

- 不添加新依赖（仅使用现有库：Tkinter, OpenCV, NumPy, Pillow, PyTorch, scikit-learn）
- 不改变核心算法逻辑
- 不添加单元测试（后续迭代）
- 不添加国际化支持
