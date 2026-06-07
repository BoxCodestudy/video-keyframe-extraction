"""CNN feature extraction using ResNet-18."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
from PIL import Image

from src.utils import estimate_eta

try:
    import torch
    from torchvision import models
    from torchvision.models import ResNet18_Weights

    HAS_ML = True
except ImportError:
    torch = None
    models = None
    ResNet18_Weights = None
    HAS_ML = False


class CNNFeatureExtractor:
    def __init__(self, batch_size: int = 16):
        if not HAS_ML:
            raise RuntimeError("当前环境未安装机器学习依赖")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        self.transform = weights.transforms()
        self.batch_size = batch_size if self.device.type == "cuda" else min(batch_size, 8)
        # Reusable thread pool (fix: was created per batch)
        self._pool = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)))

    def _load_tensor(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def extract_batch(
        self,
        image_paths: list[str],
        progress_callback: Optional[Callable[..., None]] = None,
        found_count: int = 0,
    ) -> np.ndarray:
        if not image_paths:
            return np.empty((0, 512), dtype=np.float32)
        features = []
        total = len(image_paths)
        start_time = time.time()
        for start in range(0, total, self.batch_size):
            batch_paths = image_paths[start:start + self.batch_size]
            tensors = list(self._pool.map(self._load_tensor, batch_paths))
            batch_tensor = torch.stack(tensors).to(self.device, non_blocking=self.device.type == "cuda")
            with torch.no_grad():
                batch_features = self.model(batch_tensor).detach().cpu().numpy()
            features.append(batch_features)
            current = min(start + len(batch_paths), total)
            if progress_callback:
                progress_callback(
                    stage="recognizing", current=current, total=total,
                    eta_seconds=estimate_eta(start_time, current, total),
                    found_count=found_count,
                )
        return np.concatenate(features, axis=0)

    def close(self):
        self._pool.shutdown(wait=False)
