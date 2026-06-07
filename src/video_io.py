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
