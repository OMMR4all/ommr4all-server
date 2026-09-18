"""Self-supervised ViT patch features (DINOv2 by default).

`facebook/dinov2-base` is public and ungated and is therefore the default. DINOv3
(`facebook/dinov3-vitb16-pretrain-lvd1689m`) is reachable by setting
`features.model_name` once its manual access has been granted and `HF_TOKEN` is set; the only
architectural difference this module has to know about is that DINOv3 prepends register
tokens, which is handled generically via `config.num_register_tokens`.

The image processor of the checkpoint is deliberately *not* used: `AutoImageProcessor` resizes
to 256 and centre crops to 224, which throws away most of a staff strip and destroys the
spatial feature map this package is built on.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np

from omr.discovery.config import FeatureConfig
from omr.discovery.features.base import SpatialFeatureMap, SymbolFeatureExtractor, l2_normalize

logger = logging.getLogger(__name__)


class DinoFeatureExtractor(SymbolFeatureExtractor):
    name = 'dino'

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.model = None
        self.device = 'cpu'
        self.patch_size = 14
        self.num_prefix_tokens = 1
        self.embedding_dim = 0
        self.revision: Optional[str] = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel

        if self.cfg.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.cfg.device

        logger.info('Loading %s on %s', self.cfg.model_name, self.device)
        model = AutoModel.from_pretrained(self.cfg.model_name)
        model.eval()
        model.to(self.device)
        self.model = model
        config = model.config
        self.patch_size = int(getattr(config, 'patch_size', 14))
        # DINOv3 prepends register tokens after the class token; DINOv2 has none.
        self.num_prefix_tokens = 1 + int(getattr(config, 'num_register_tokens', 0) or 0)
        self.embedding_dim = int(getattr(config, 'hidden_size', 0))
        self.revision = getattr(config, '_commit_hash', None)

    def _preprocess(self, image: np.ndarray, staff_space_px: float):
        import cv2
        import torch

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        crop_h, crop_w = image.shape[:2]

        scale = 1.0
        if staff_space_px > 0 and self.cfg.target_staff_space_px > 0:
            scale = float(self.cfg.target_staff_space_px) / float(staff_space_px)
        # never shrink below one patch, never exceed the input budget
        scale = max(scale, self.patch_size / max(1.0, min(crop_h, crop_w)))
        longest = max(crop_h, crop_w) * scale
        if longest > self.cfg.max_input_side:
            scale *= self.cfg.max_input_side / longest

        new_h = max(self.patch_size, int(round(crop_h * scale)))
        new_w = max(self.patch_size, int(round(crop_w * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA
                             if scale < 1.0 else cv2.INTER_LINEAR)

        pad_h = (-new_h) % self.patch_size
        pad_w = (-new_w) % self.patch_size
        if pad_h or pad_w:
            border = np.median(np.concatenate([
                resized[0].reshape(-1, 3), resized[-1].reshape(-1, 3),
                resized[:, 0].reshape(-1, 3), resized[:, -1].reshape(-1, 3)]), axis=0)
            padded = np.empty((new_h + pad_h, new_w + pad_w, 3), dtype=resized.dtype)
            padded[:] = border.astype(resized.dtype)
            padded[:new_h, :new_w] = resized
            resized = padded

        tensor = torch.from_numpy(np.ascontiguousarray(resized)).float().div_(255.0)
        mean = torch.tensor(self.cfg.normalize_mean, dtype=torch.float32).view(1, 1, 3)
        std = torch.tensor(self.cfg.normalize_std, dtype=torch.float32).view(1, 1, 3)
        tensor = (tensor - mean) / std
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        content_size = (new_h, new_w)
        return tensor, content_size, (crop_h, crop_w), (resized.shape[0], resized.shape[1])

    def extract_feature_map(self, image: np.ndarray, staff_space_px: float) -> SpatialFeatureMap:
        import torch
        self.load()
        tensor, content_size, crop_size, input_size = self._preprocess(image, staff_space_px)
        with torch.inference_mode():
            hidden = self.model(tensor).last_hidden_state
        tokens = hidden[0, self.num_prefix_tokens:].to('cpu').numpy().astype(np.float32)
        gh = input_size[0] // self.patch_size
        gw = input_size[1] // self.patch_size
        if tokens.shape[0] != gh * gw:
            raise RuntimeError('{} returned {} patch tokens for a {}x{} grid'.format(
                self.cfg.model_name, tokens.shape[0], gh, gw))
        features = l2_normalize(tokens.reshape(gh, gw, -1))
        return SpatialFeatureMap(features=features, patch_size=self.patch_size,
                                 input_size=input_size, content_size=content_size,
                                 crop_size=crop_size)

    def describe(self) -> Dict[str, Any]:
        self.load()
        import torch
        return {
            'backend': self.name,
            'model_name': self.cfg.model_name,
            'model_revision': self.revision,
            'patch_size': self.patch_size,
            'embedding_dim': self.embedding_dim,
            'num_prefix_tokens': self.num_prefix_tokens,
            'device': self.device,
            'dtype': str(next(self.model.parameters()).dtype) if self.model is not None else None,
            'target_staff_space_px': self.cfg.target_staff_space_px,
            'max_input_side': self.cfg.max_input_side,
            'normalize_mean': list(self.cfg.normalize_mean),
            'normalize_std': list(self.cfg.normalize_std),
            'torch_version': torch.__version__,
        }
