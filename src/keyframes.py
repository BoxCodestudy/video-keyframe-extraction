"""Keyframe selection via KMeans clustering and anchor-based similarity filtering."""
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import AnalysisResult, IntrusionEvent, VideoMetadata, DEFAULT_KEYFRAME_COUNT, DEFAULT_SIMILARITY_THRESHOLD
from src.features import HAS_ML, CNNFeatureExtractor
from src.utils import extract_frame_index, list_image_files

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


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
    if not HAS_ML or KMeans is None:
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
    kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
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
