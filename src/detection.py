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
        # Performance optimization: skip HOG result for near-static frames
        motion_signal_raw = float(np.clip((motion_ratio - 0.015) / 0.12, 0.0, 1.0))
        diff_signal_raw = float(np.clip((diff_ratio - 0.01) / 0.08, 0.0, 1.0))
        if motion_signal_raw < 0.03 and diff_signal_raw < 0.03:
            person_signal = 0.0
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
