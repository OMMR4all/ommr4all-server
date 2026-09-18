"""The feature extractor interface.

A feature extractor turns a staff crop into a dense grid of patch descriptors. Everything
downstream (discovery by graph cut, candidate embeddings, neume appearance) reads only
`SpatialFeatureMap`, so a backbone can be swapped without touching any other stage.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


class SpatialFeatureMap(NamedTuple):
    """Patch descriptors plus exact crop, transformed-content, and padded-input geometry."""
    features: np.ndarray  # (gh, gw, D) float32
    patch_size: int
    input_size: Tuple[int, int]  # (h, w) of the padded tensor fed to the backbone
    content_size: Tuple[int, int]  # (h, w) after transform, before bottom/right padding
    crop_size: Tuple[int, int]  # (h, w) of the original crop

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.features.shape[0], self.features.shape[1]

    @property
    def dim(self) -> int:
        return self.features.shape[2]

    def _validate_geometry(self) -> None:
        if self.features.ndim != 3:
            raise ValueError('features must have shape (grid_height, grid_width, dim)')
        if self.patch_size <= 0:
            raise ValueError('patch_size must be positive')
        sizes = (self.crop_size, self.content_size, self.input_size)
        if any(len(size) != 2 for size in sizes):
            raise ValueError('crop_size, content_size, and input_size must be two-dimensional')
        if any(value <= 0 for size in sizes for value in size):
            raise ValueError('crop, content, and input dimensions must be positive')
        if (self.content_size[0] > self.input_size[0]
                or self.content_size[1] > self.input_size[1]):
            raise ValueError('content_size must not exceed input_size')
        gh, gw = self.grid_size
        expected = (gh * self.patch_size, gw * self.patch_size)
        if self.input_size != expected:
            raise ValueError('input_size {} does not match grid geometry {}'.format(
                self.input_size, expected))

    @property
    def scale_xy(self) -> Tuple[float, float]:
        """Independent transformed-content pixels per crop pixel in (x, y) order."""
        self._validate_geometry()
        crop_h, crop_w = self.crop_size
        content_h, content_w = self.content_size
        return content_w / crop_w, content_h / crop_h

    def crop_pixels_per_patch(self) -> Tuple[float, float]:
        """Independent crop-pixel extents of one complete patch in (x, y) order."""
        scale_x, scale_y = self.scale_xy
        return self.patch_size / scale_x, self.patch_size / scale_y

    def patch_center(self, row: int, col: int) -> Tuple[float, float]:
        """Crop-pixel centre of a patch's intersection with the unpadded content."""
        self._validate_geometry()
        gh, gw = self.grid_size
        if row < 0 or row >= gh or col < 0 or col >= gw:
            raise IndexError('patch cell ({}, {}) is outside grid {}'.format(row, col,
                                                                             self.grid_size))
        content_h, content_w = self.content_size
        scale_x, scale_y = self.scale_xy
        x0 = min(col * self.patch_size, content_w)
        x1 = min((col + 1) * self.patch_size, content_w)
        y0 = min(row * self.patch_size, content_h)
        y1 = min((row + 1) * self.patch_size, content_h)
        return (x0 + x1) / (2.0 * scale_x), (y0 + y1) / (2.0 * scale_y)

    def area_density(self, mask: np.ndarray) -> np.ndarray:
        """Area-average a crop-pixel mask onto exact padded patch cells."""
        import cv2

        self._validate_geometry()
        if mask.ndim < 2 or mask.shape[:2] != self.crop_size:
            raise ValueError('mask shape {} does not match crop_size {}'.format(
                mask.shape[:2], self.crop_size))
        content_h, content_w = self.content_size
        input_h, input_w = self.input_size
        resized = cv2.resize(mask.astype(np.float32), (content_w, content_h),
                             interpolation=cv2.INTER_AREA)
        padded = np.zeros((input_h, input_w), dtype=np.float32)
        padded[:content_h, :content_w] = resized
        gh, gw = self.grid_size
        ps = self.patch_size
        return padded.reshape(gh, ps, gw, ps).mean(axis=(1, 3))

    def grid_to_crop(self, values: np.ndarray, *, nearest: bool = False) -> np.ndarray:
        """Project a scalar patch grid through padded input and content back to the crop."""
        import cv2

        self._validate_geometry()
        if values.ndim != 2 or values.shape != self.grid_size:
            raise ValueError('grid values shape {} does not match grid_size {}'.format(
                values.shape, self.grid_size))
        input_h, input_w = self.input_size
        content_h, content_w = self.content_size
        crop_h, crop_w = self.crop_size
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        projected = cv2.resize(values.astype(np.float32), (input_w, input_h),
                               interpolation=interpolation)
        content = projected[:content_h, :content_w]
        return cv2.resize(content, (crop_w, crop_h), interpolation=interpolation)

    def patch_of_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Crop pixel -> (row, col) of the patch that contains it, clipped to the grid."""
        self._validate_geometry()
        gh, gw = self.grid_size
        scale_x, scale_y = self.scale_xy
        row = int(np.clip(int((y * scale_y) // self.patch_size), 0, gh - 1))
        col = int(np.clip(int((x * scale_x) // self.patch_size), 0, gw - 1))
        return row, col

    def _bilinear(self, x: float, y: float) -> np.ndarray:
        self._validate_geometry()
        gh, gw = self.grid_size
        scale_x, scale_y = self.scale_xy
        fy = y * scale_y / self.patch_size - 0.5
        fx = x * scale_x / self.patch_size - 0.5
        y0 = int(np.floor(np.clip(fy, 0, gh - 1)))
        x0 = int(np.floor(np.clip(fx, 0, gw - 1)))
        y1 = min(y0 + 1, gh - 1)
        x1 = min(x0 + 1, gw - 1)
        wy = float(np.clip(fy - y0, 0.0, 1.0))
        wx = float(np.clip(fx - x0, 0.0, 1.0))
        f = self.features
        top = f[y0, x0] * (1 - wx) + f[y0, x1] * wx
        bottom = f[y1, x0] * (1 - wx) + f[y1, x1] * wx
        return top * (1 - wy) + bottom * wy

    def pool_box(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        """Average the patch descriptors overlapping a box given in crop pixels."""
        self._validate_geometry()
        gh, gw = self.grid_size
        scale_x, scale_y = self.scale_xy
        ps = self.patch_size
        r0 = y * scale_y / ps
        r1 = (y + h) * scale_y / ps
        c0 = x * scale_x / ps
        c1 = (x + w) * scale_x / ps
        if (r1 - r0) < 1.0 or (c1 - c0) < 1.0:
            return l2_normalize(self._bilinear(x + w / 2.0, y + h / 2.0))

        row_lo = int(np.clip(np.floor(r0), 0, gh - 1))
        row_hi = int(np.clip(np.ceil(r1), 1, gh))
        col_lo = int(np.clip(np.floor(c0), 0, gw - 1))
        col_hi = int(np.clip(np.ceil(c1), 1, gw))
        row_hi = max(row_hi, row_lo + 1)
        col_hi = max(col_hi, col_lo + 1)
        patch = self.features[row_lo:row_hi, col_lo:col_hi]
        return l2_normalize(patch.reshape(-1, patch.shape[-1]).mean(axis=0))


class SymbolFeatureExtractor(ABC):
    """Base class of every backbone. Heavy imports and weight loading happen in `load()`."""
    name: str = 'base'

    def load(self) -> None:
        """Prepare the backbone. Called once before the first `extract_feature_map`."""

    @abstractmethod
    def extract_feature_map(self, image: np.ndarray, staff_space_px: float) -> SpatialFeatureMap:
        ...

    def extract_embedding(self, image: np.ndarray, staff_space_px: float) -> np.ndarray:
        feature_map = self.extract_feature_map(image, staff_space_px)
        h, w = image.shape[:2]
        return feature_map.pool_box(0, 0, w, h)

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Everything about this extractor that belongs into the run record."""
