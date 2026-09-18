"""Configuration of a discovery run.

Every knob of every stage lives here so that a run is fully described by `RunConfig`, which
is stored verbatim in `run.json`. `config_hash()` is part of the run id, so two runs with
different parameters can never collide in the output directory.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class FeatureConfig(DataClassJSONMixin):
    """Patch-level feature extractor.

    `target_staff_space_px` is the scale prior of the whole package: a staff crop is rescaled
    so that one staff space measures this many pixels before it is fed to the backbone. With
    the DINOv2 patch size of 14 the default gives three patch tokens per staff space, i.e.
    roughly 3x3 tokens for a punctum.
    """
    backend: str = 'dino'  # key into omr.discovery.features.FEATURE_EXTRACTORS
    model_name: str = 'facebook/dinov2-base'
    device: str = 'auto'  # 'auto' | 'cpu' | 'cuda'
    target_staff_space_px: float = 42.0
    max_input_side: int = 2048
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class CropConfig(DataClassJSONMixin):
    """Staff crop extraction, see `omr.discovery.regions`.

    `dewarp` must stay False: `ImageExtractDewarpedStaffLineImages.global_to_local_pos`
    applies the same forward transform as `local_to_global_pos` instead of its inverse, so
    with dewarping enabled the page -> crop direction is wrong and every candidate box would
    be placed incorrectly on the page.

    `center` must stay False as well: `_resize_to_height` of the same operation pads with a
    two dimensional block, which cannot be stacked onto the three channel colour crop that
    `SymbolDetectionDatasetTorch` produces -- every line would raise and be silently dropped
    by `Dataset._load`. The dataset already normalises the vertical extent to the staff line
    bounding box plus one average line distance above and below.
    """
    crop_height: int = 180
    pad: List[int] = field(default_factory=lambda: [0, 10, 0, 40])  # (top, right, bottom, left)
    dewarp: bool = False
    center: bool = False
    staff_lines_only: bool = True
    #: Ink inside the lines of non-music blocks (lyrics, drop capitals, paragraphs) is
    #: removed. A staff crop reaches one average line distance below the lowest staff line,
    #: which on a densely written page already contains the lyrics line -- without this the
    #: text dominates the candidate set.
    mask_non_music_blocks: bool = True
    #: Ink outside the music line's own polygon (dilated, see below) is removed as well.
    #: Falls back to "no restriction" when the line carries no polygon.
    restrict_to_line_polygon: bool = True
    line_polygon_dilate_staff_space: float = 0.5


@dataclass
class DiscoveryConfig(DataClassJSONMixin):
    """Class-agnostic symbol localisation.

    All size thresholds are expressed in staff spaces, never in pixels, so the same
    configuration transfers between books of different resolutions.
    """
    method: str = 'ink+tokencut'  # '+'-separated subset of SYMBOL_DISCOVERY_METHODS
    min_area_staff_space_sq: float = 0.0625
    #: A region without ink is not a symbol, whichever method proposed it.
    min_ink_area_staff_space_sq: float = 0.03
    max_height_staff_space: float = 6.0
    max_width_staff_space: float = 1.6
    split_min_distance_staff_space: float = 0.7
    #: Split vertically stacked, touching noteheads using strong distance-transform peaks.
    #: The conservative shape and left-margin gates avoid splitting tall clefs and virgae.
    stacked_split_min_height_staff_space: float = 1.3
    stacked_split_max_width_staff_space: float = 1.35
    stacked_split_min_peak_radius_staff_space: float = 0.14
    stacked_split_min_peak_dy_staff_space: float = 0.45
    stacked_split_max_peak_dy_staff_space: float = 1.35
    stacked_split_max_peak_dx_staff_space: float = 0.45
    stacked_split_ignore_left_staff_space: float = 1.0
    box_pad_staff_space: float = 0.15
    staff_line_removal_max_run_staff_space: float = 0.35
    #: A staff line is thin *and* long. Ink is only treated as a staff line when its vertical
    #: run is short and it either runs horizontally for at least this many staff spaces or
    #: sits on an annotated staff line.
    staff_line_removal_min_length_staff_space: float = 1.5
    close_kernel_staff_space: float = 0.2
    tokencut_tau: float = 0.2
    tokencut_eps: float = 1e-5
    #: one object per cut (the connected component holding the most salient patch), so the
    #: number of cuts bounds how many symbols a window can yield
    tokencut_n_cuts: int = 4
    # A staff crop easily carries a few thousand patch tokens and the dense eigen
    # decomposition of the normalised Laplacian is O(n^3), so the crop is processed in
    # overlapping column windows. The window is deliberately only a few staff spaces wide so
    # that `tokencut_n_cuts` is of the order of the number of symbols inside it. The overlap
    # is resolved by the global NMS.
    tokencut_window_staff_space: float = 4.0
    tokencut_window_overlap_staff_space: float = 1.0
    tokencut_max_nodes: int = 1500
    patch_peak_background_max_ink_density: float = 0.01
    patch_peak_local_background_staff_space: float = 2.0
    patch_peak_min_background_patches: int = 8
    patch_peak_min_ink_density: float = 0.05
    patch_peak_min_score: float = 0.05
    patch_peak_min_distance_staff_space: float = 0.5
    patch_peak_prominence_staff_space: float = 0.75
    patch_peak_min_prominence: float = 0.01
    patch_peak_box_staff_space: float = 1.0
    nms_iou: float = 0.4


@dataclass
class ClusterConfig(DataClassJSONMixin):
    """Appearance clustering. Cluster ids are similarity labels, never semantic classes."""
    method: str = 'hdbscan'  # key into omr.discovery.clustering.CLUSTERERS
    pca_dim: Optional[int] = 64
    geometry_weight: float = 0.0
    min_cluster_size: int = 8
    n_clusters: int = 40
    distance_threshold: Optional[float] = None
    seed: int = 0


@dataclass
class GroupingConfig(DataClassJSONMixin):
    """Grouping of symbols into neumes."""
    method: str = 'rule_graph'  # key into omr.discovery.grouping.NEUME_GROUPING_METHODS
    max_dx_staff_space: float = 0.7
    min_gap_ink: float = 0.6
    geometry_only: bool = False  # ablation: ignore the image evidence between two symbols
    accept_unknown_family: bool = True  # group candidates whose family is still UNKNOWN


@dataclass
class NeumeEmbeddingConfig(DataClassJSONMixin):
    appearance_weight: float = 1.0
    structure_weight: float = 1.0


@dataclass
class EvaluationConfig(DataClassJSONMixin):
    """PCGTS stores a single centre per symbol and no bounding box, so localisation is
    evaluated by centre distance. `derived_gt_box_staff_space` is only used to synthesise a
    pseudo box, and every number derived from it is reported under a `_derived_gt_box` key.
    """
    match_radius_staff_space: float = 0.5
    derived_gt_box_staff_space: float = 1.4
    retrieval_k: List[int] = field(default_factory=lambda: [1, 5])


@dataclass
class RunConfig(DataClassJSONMixin):
    book: str = 'demo'
    pages: List[str] = field(default_factory=list)  # empty == every page of the book
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    neume_embedding: NeumeEmbeddingConfig = field(default_factory=NeumeEmbeddingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Used to layer a `--config` file and then the command line flags over the defaults.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict] = None) -> RunConfig:
    merged = RunConfig().to_dict()
    if path:
        with open(path) as f:
            merged = deep_merge(merged, json.load(f))
    if overrides:
        merged = deep_merge(merged, overrides)
    return RunConfig.from_dict(merged)
