"""Embedding-based diverse page selection from staff foreground or whole-page images."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from mashumaro.mixins.json import DataClassJSONMixin

from omr.discovery.config import DiscoveryConfig, PageSuggestionConfig
from omr.discovery.discovery import remove_staff_lines
from omr.discovery.features import l2_normalize


SUGGESTIONS_FILE = 'page_suggestions.json'
PAGE_EMBEDDINGS_FILE = 'page_embeddings.npy'
PAGE_EMBEDDING_INDEX_FILE = 'page_embeddings.json'


@dataclass
class PageEmbeddingStats(DataClassJSONMixin):
    page: str
    n_music_lines: int
    n_foreground_patches: int
    foreground_weight: float


@dataclass
class PageSuggestion(DataClassJSONMixin):
    rank: int
    page: str
    novelty_score: float
    nearest_reference_page: Optional[str]
    reason: str
    n_music_lines: int = 0
    n_foreground_patches: int = 0


def _pool_weighted_moments(patches) -> Tuple[Optional[np.ndarray], int, float]:
    weighted_sum = None
    weighted_square_sum = None
    total_weight = 0.0
    n_patches = 0
    for feature_map, weights in patches:
        active = weights.reshape(-1) > 0
        if not np.any(active):
            continue
        features = feature_map.features.reshape(-1, feature_map.dim)[active].astype(np.float64)
        active_weights = weights.reshape(-1)[active].astype(np.float64)
        if weighted_sum is None:
            weighted_sum = np.zeros(features.shape[1], dtype=np.float64)
            weighted_square_sum = np.zeros(features.shape[1], dtype=np.float64)
        weighted_sum += np.sum(features * active_weights[:, None], axis=0)
        weighted_square_sum += np.sum(np.square(features) * active_weights[:, None], axis=0)
        total_weight += float(np.sum(active_weights))
        n_patches += int(np.count_nonzero(active))

    if weighted_sum is None or total_weight <= 0:
        return None, n_patches, total_weight
    mean = weighted_sum / total_weight
    variance = np.maximum(weighted_square_sum / total_weight - np.square(mean), 0.0)
    descriptor = np.concatenate((mean, np.sqrt(variance))).astype(np.float32)
    return l2_normalize(descriptor).astype(np.float32), n_patches, total_weight


def foreground_page_embedding(crops, extractor, cfg: PageSuggestionConfig,
                              discovery_cfg: DiscoveryConfig) -> Tuple[Optional[np.ndarray], PageEmbeddingStats]:
    """Pool foreground-weighted patch moments across staff crops."""
    def patches():
        for crop in crops:
            feature_map = extractor.extract_feature_map(crop.image, crop.staff_space_px)
            foreground = remove_staff_lines(crop.binary, crop.staff_lines_px, crop.staff_space_px,
                                            discovery_cfg)
            weights = feature_map.area_density(foreground & crop.region_mask)
            weights[weights < cfg.min_ink_density] = 0.0
            yield feature_map, weights

    descriptor, n_patches, total_weight = _pool_weighted_moments(patches())
    return descriptor, PageEmbeddingStats(crops[0].page if crops else '', len(crops),
                                          n_patches, total_weight)


class UnreadableOriginalImage(OSError):
    """Image decoding failed; inference failures must not be mistaken for bad images."""


def whole_image_page_embedding(page, extractor) -> np.ndarray:
    """Describe an original image, without reading PCGTS or staff geometry."""
    try:
        with Image.open(page.file('color_original').local_path()) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((448, 448), Image.Resampling.LANCZOS)
            pixels = np.asarray(image)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise UnreadableOriginalImage(str(exc)) from exc
    feature_map = extractor.extract_feature_map(pixels, 0)
    weights = feature_map.area_density(np.ones(pixels.shape[:2], dtype=np.uint8))
    descriptor, _, _ = _pool_weighted_moments(((feature_map, weights),))
    if descriptor is None:
        raise ValueError('original image has no valid feature patches')
    return descriptor


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    # Inputs are normalised when produced, but normalising here keeps the pure selection API safe
    # for callers that load or construct embeddings independently.
    left = l2_normalize(np.asarray(left, dtype=np.float32))
    right = l2_normalize(np.asarray(right, dtype=np.float32))
    return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))


def select_representative_pages(embeddings: Mapping[str, np.ndarray],
                                candidate_pages: Iterable[str],
                                corrected_pages: Iterable[str],
                                count: int) -> List[PageSuggestion]:
    """Select a deterministic embedding core-set with farthest-first traversal.

    Corrected pages seed the reference set. Without corrected pages, the first suggestion is the
    page nearest the candidate centroid rather than an arbitrary outlier; subsequent suggestions
    maximise their minimum cosine distance to corrected or already selected pages.
    """
    if count < 1:
        raise ValueError('page suggestion count must be positive')

    corrected = sorted({name for name in corrected_pages if name in embeddings})
    candidates = sorted({name for name in candidate_pages
                         if name in embeddings and name not in corrected})
    if not candidates:
        return []

    dimensions = {np.asarray(embeddings[name]).shape for name in corrected + candidates}
    if len(dimensions) != 1:
        raise ValueError('all page embeddings must have the same shape')

    selected: List[PageSuggestion] = []
    references = list(corrected)
    remaining = set(candidates)

    if not references:
        matrix = np.stack([l2_normalize(np.asarray(embeddings[name], dtype=np.float32))
                           for name in candidates])
        centroid = l2_normalize(matrix.mean(axis=0))
        first = min(candidates, key=lambda name: (_cosine_distance(embeddings[name], centroid),
                                                  name))
        selected.append(PageSuggestion(1, first, 0.0, None, 'representative_seed'))
        references.append(first)
        remaining.remove(first)

    while remaining and len(selected) < min(count, len(candidates)):
        scored: Dict[str, Tuple[float, str]] = {}
        for name in remaining:
            nearest = min(references,
                          key=lambda reference: (_cosine_distance(embeddings[name],
                                                                  embeddings[reference]),
                                                 reference))
            scored[name] = (_cosine_distance(embeddings[name], embeddings[nearest]), nearest)
        page = min(remaining, key=lambda name: (-scored[name][0], name))
        score, nearest = scored[page]
        selected.append(PageSuggestion(len(selected) + 1, page, score, nearest,
                                       'farthest_from_reference'))
        references.append(page)
        remaining.remove(page)

    return selected
