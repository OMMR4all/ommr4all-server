"""Reproducibility record of a run: environment, versions, seeds, timings, peak memory."""
import logging
import os
import platform
import random
import resource
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mashumaro.mixins.json import DataClassJSONMixin

from omr.discovery.config import RunConfig

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed every source of randomness the package can reach.

    The seed is additionally handed to scikit-learn as `random_state`, because a global seed
    does not reach its own RNGs.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception as e:  # pragma: no cover - numpy is a hard dependency
        logger.warning('Could not seed numpy: %s', e)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception as e:
        logger.info('Could not seed torch (not installed or unavailable): %s', e)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'],
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             capture_output=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except Exception as e:
        logger.info('Could not determine the git commit: %s', e)
    return None


def _version(module_name: str, attribute: str = '__version__') -> Optional[str]:
    try:
        module = __import__(module_name)
        return str(getattr(module, attribute))
    except Exception:
        return None


def collect_environment() -> Dict[str, Any]:
    return {
        'git_commit': _git_commit(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'torch_version': _version('torch'),
        'transformers_version': _version('transformers'),
        'sklearn_version': _version('sklearn'),
        'numpy_version': _version('numpy'),
        'scipy_version': _version('scipy'),
        'opencv_version': _version('cv2'),
    }


def peak_rss_bytes() -> int:
    """Peak resident set size of this process. `ru_maxrss` is in kilobytes on Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def peak_cuda_bytes() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


@dataclass
class RunRecord(DataClassJSONMixin):
    """Everything needed to repeat a run and to judge its cost."""
    schema_version: int
    run_id: str
    kind: str  # 'symbols' | 'neumes' | 'page_suggestions'
    config: RunConfig
    created_at: str = ''
    finished_at: str = ''
    parent_run_id: Optional[str] = None
    symbol_source: Optional[str] = None  # neume runs: 'discovered' | 'groundtruth'
    environment: Dict[str, Any] = field(default_factory=dict)
    device: str = ''
    feature_extractor: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    pages: List[str] = field(default_factory=list)
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    peak_rss_bytes: int = 0
    peak_cuda_bytes: Optional[int] = None
    n_lines_total: int = 0
    n_lines_dropped: int = 0
    notes: List[str] = field(default_factory=list)


class StageTimer:
    """Accumulates wall clock seconds per stage into a `RunRecord`."""

    def __init__(self, record: RunRecord):
        self.record = record

    def __call__(self, stage: str) -> '_StageContext':
        return _StageContext(self.record, stage)


class _StageContext:
    def __init__(self, record: RunRecord, stage: str):
        self.record = record
        self.stage = stage
        self.t0 = 0.0

    def __enter__(self) -> '_StageContext':
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self.t0
        self.record.stage_seconds[self.stage] = self.record.stage_seconds.get(self.stage, 0.0) + elapsed
        return False
