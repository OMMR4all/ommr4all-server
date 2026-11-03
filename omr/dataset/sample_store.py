"""Sample collectors that keep a training dataset in memory while it fits and write it to the
scratch directory of a `DatasetSpill` once it does not.

Two shapes of sample are covered:

* `ImageMaskStore` — (image, mask, per-head masks) for the segmentation trainers. Its
  `to_dataframe()` has the columns the `segmentation` library expects; in memory mode the
  values are numpy arrays (read by `segmentation.datasets.dataset.MemoryDataset`), on disk
  they are PNG paths (read by `MaskDataset`, which is otherwise identical).
* `BlobStore` — encoded images (end2end JPEG crops). `add()` hands out a mutable `Blob` that
  reads from memory or from a file, so already collected samples can be flushed later.

`write_line_samples` is the third case: the OCR trainers write every sample into a temp
directory for the training library anyway, so they only need to stream.
"""
import csv
import io
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from omr.dataset.spill import DatasetSpill

logger = logging.getLogger(__name__)


def to_image_array(arr: np.ndarray, what: str) -> np.ndarray:
    """`arr` in a dtype PNG can store without losing values.

    All image operations produce uint8 images and uint8 label masks; anything else would
    silently change what the network is trained on, so it is rejected instead."""
    if arr.dtype == np.uint8 or arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.integer) and arr.size and 0 <= arr.min() and arr.max() < 256:
        return arr.astype(np.uint8)
    raise RuntimeError(f"Cannot write {what} of dtype {arr.dtype} to disk without losing information. "
                       f"Raise OMMR4ALL_DATASET_MEMORY_BUDGET_MB to keep the dataset in memory.")


def write_image(path: str, arr: np.ndarray, what: str):
    # compress_level 1: masks still shrink by an order of magnitude, and the dataset is
    # written while the GPU is idle, so time matters more than the last few percent
    Image.fromarray(to_image_array(arr, what)).save(path, compress_level=1)


def nbytes(arr: Optional[np.ndarray]) -> int:
    return 0 if arr is None else int(getattr(arr, 'nbytes', 0))


@dataclass
class Blob:
    """An encoded image, either in memory or in a file. Mutable so that a `BlobStore` can
    move it to disk after it has been handed out."""
    data: Optional[bytes] = None
    path: Optional[str] = None

    def read(self) -> bytes:
        if self.data is not None:
            return self.data
        with open(self.path, 'rb') as f:
            return f.read()

    def open(self) -> BinaryIO:
        if self.data is not None:
            return io.BytesIO(self.data)
        return open(self.path, 'rb')


class BlobStore:
    """Collects encoded images; spills them to files once the budget is exhausted."""

    def __init__(self, spill: Optional[DatasetSpill] = None, suffix: str = '.jpg'):
        self.spill = spill
        self.suffix = suffix
        self._blobs: List[Blob] = []

    def add(self, data: bytes) -> Blob:
        if self.spill is None:
            return Blob(data=data)
        if self.spill.account(len(data)):
            # also moves everything collected before the budget ran out
            self._flush()
            return Blob(path=self._write(data))

        blob = Blob(data=data)
        self._blobs.append(blob)
        return blob

    def _write(self, data: bytes) -> str:
        path = self.spill.path(self.suffix)
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def _flush(self):
        """Writes the blobs collected before the budget ran out."""
        if not self._blobs:
            return
        logger.info(f"Writing {len(self._blobs)} already encoded samples to disk")
        for blob in self._blobs:
            if blob.data is not None:
                blob.path = self._write(blob.data)
                blob.data = None
        self._blobs = []


class ImageMaskStore:
    """Collects (image, mask, additional masks) samples for the segmentation trainers.

    `to_dataframe()` returns the DataFrame the trainers pass to `MemoryDataset` (arrays) or,
    when `spilled`, to `MaskDataset` (PNG paths)."""

    def __init__(self, spill: Optional[DatasetSpill] = None, head_count: int = 0,
                 keep_original: bool = True):
        self.spill = spill
        self.head_count = head_count
        self.keep_original = keep_original
        self.images: List[Any] = []
        self.masks: List[Any] = []
        self.originals: List[Any] = []
        self.add_masks: List[List[Any]] = [[] for _ in range(head_count)]

    @property
    def spilled(self) -> bool:
        return self.spill is not None and self.spill.spilled

    def __len__(self):
        return len(self.images)

    def add(self, image: np.ndarray, mask: Optional[np.ndarray],
            additional_masks: Iterable[Optional[np.ndarray]] = (), original: Any = None):
        additional_masks = list(additional_masks)
        if self.spill is not None:
            self.spill.account(nbytes(image) + nbytes(mask) + sum(nbytes(m) for m in additional_masks))

        if self.spilled:
            self._flush()
            image = self._write(image, 'image')
            mask = self._write(mask, 'mask')
            additional_masks = [self._write(m, 'mask') for m in additional_masks]

        self.images.append(image)
        self.masks.append(mask)
        self.originals.append(original if self.keep_original else None)
        for i in range(self.head_count):
            self.add_masks[i].append(additional_masks[i] if i < len(additional_masks) else None)

    def _write(self, arr: Optional[np.ndarray], what: str) -> Optional[str]:
        if arr is None:
            return None
        path = self.spill.path('.png')
        write_image(path, arr, what)
        return path

    def _flush(self):
        """Writes the samples collected before the budget ran out."""
        if not self.images or isinstance(self.images[0], str):
            return
        logger.info(f"Writing {len(self.images)} already loaded samples to disk")
        self.images = [self._write(i, 'image') for i in self.images]
        self.masks = [self._write(m, 'mask') for m in self.masks]
        for head in self.add_masks:
            head[:] = [self._write(m, 'mask') for m in head]

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(data={'images': self.images, 'masks': self.masks,
                                  'original': self.originals,
                                  **{f'add_mask_{i}': self.add_masks[i] for i in range(self.head_count)}})


def write_line_samples(path: str, samples: Iterable[Tuple[np.ndarray, str]],
                       unique_names: bool = False) -> int:
    """Writes `(image, gt)` samples as `<name>.png` + `<name>.txt` plus the `labels.csv` the
    OCR training libraries read, and returns the number of samples written.

    Consumes `samples` lazily, so only one line image is in memory at a time."""
    os.makedirs(path, exist_ok=True)
    written = 0
    with open(os.path.join(path, "labels.csv"), 'w', encoding='UTF8') as labels_csv:
        csv_writer = csv.writer(labels_csv)
        csv_writer.writerow(["filename", "words"])
        for ind, (image, gt) in enumerate(samples):
            name = f"{ind}{uuid.uuid4()}" if unique_names else str(ind)
            Image.fromarray(to_image_array(image, 'line image')).save(os.path.join(path, name + ".png"))
            with open(os.path.join(path, name + ".txt"), 'w', encoding='UTF8') as gt_txt:
                gt_txt.write(gt)
            csv_writer.writerow([name + ".png", gt])
            written += 1
    return written
