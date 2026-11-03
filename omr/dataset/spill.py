"""Memory budget and scratch directory for datasets that are built at training time.

Training datasets are generated from the PcGts files on every run and kept in memory, which
is fast but bounded by the available RAM. A `DatasetSpill` accounts the bytes the sample
stores (`omr.dataset.sample_store`) collect and, once the budget is exhausted, hands them a
temporary directory to write the samples into instead. Everything collected so far is
flushed as well, so the memory is actually reclaimed.

Configuration:
    OMMR4ALL_DATASET_MEMORY_BUDGET_MB   bytes to keep in RAM before spilling; 0 spills
                                        immediately (useful to test the disk path).
                                        Default: 40 % of the RAM free when the spill is
                                        created, at least MIN_BUDGET_BYTES.
    OMMR4ALL_DATASET_CACHE_DIR          where to write spilled samples. Default: the
                                        system temp dir (honours TMPDIR).
"""
import logging
import os
import shutil
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

MIN_BUDGET_BYTES = 512 * 1024 ** 2
# fraction of the currently free RAM a dataset may occupy by default
DEFAULT_BUDGET_FRACTION = 0.4
# spill as soon as the machine has less than this much RAM free, whatever the budget says:
# several trainings may run at once, so the free memory at startup is not a reservation
MIN_FREE_BYTES = 1024 ** 3
# how many `account` calls to wait between two free-memory checks
FREE_MEMORY_CHECK_INTERVAL = 50

# DataLoader workers pass every batch through /dev/shm (Docker default: 64 MB)
MIN_SHM_BYTES = 1 << 30


def loader_workers(requested: int) -> int:
    """`requested` data loader workers, or 0 if /dev/shm is too small to hold their batches;
    running out of shm crashes training with "unable to allocate shared memory"."""
    if requested <= 0:
        return 0
    try:
        free = shutil.disk_usage('/dev/shm').free
    except OSError:
        return requested
    if free < MIN_SHM_BYTES:
        logger.warning(f"Only {free >> 20} MB free in /dev/shm, loading data in the main process. "
                       f"Give the container more (docker: shm_size / --shm-size) to use {requested} workers.")
        return 0
    return requested


def available_memory() -> Optional[int]:
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception as e:  # psutil is a hard dependency, but never fail a training over this
        logger.debug(f"Could not determine the available memory: {e}")
        return None


def default_budget_bytes() -> int:
    """The configured budget, or a fraction of the currently free RAM."""
    configured = (os.environ.get('OMMR4ALL_DATASET_MEMORY_BUDGET_MB') or '').strip()
    if configured:
        try:
            return max(0, int(float(configured) * 1024 ** 2))
        except ValueError:
            logger.warning(f"Ignoring OMMR4ALL_DATASET_MEMORY_BUDGET_MB={configured!r}, not a number")

    available = available_memory()
    if available is None:
        return MIN_BUDGET_BYTES
    return max(MIN_BUDGET_BYTES, int(available * DEFAULT_BUDGET_FRACTION))


class DatasetSpill:
    """Shared byte budget and lazily created scratch directory for one training run.

    `account(n)` adds `n` bytes to the running total and returns True once the dataset must
    go to disk; from then on it stays True (`spilled`). The directory is only created on the
    first `path()` call and removed by `close()`, so runs that fit in memory never touch the
    disk.
    """

    def __init__(self, budget_bytes: Optional[int] = None, root: Optional[str] = None):
        self.budget_bytes = default_budget_bytes() if budget_bytes is None else budget_bytes
        self.root = root or (os.environ.get('OMMR4ALL_DATASET_CACHE_DIR') or '').strip() or None
        self.bytes = 0
        self.samples = 0
        self.spilled = False
        self._dir: Optional[str] = None
        self._files = 0
        self._checks = 0
        logger.info(f"Dataset memory budget: {self.budget_bytes >> 20} MB")

    @property
    def dir(self) -> Optional[str]:
        return self._dir

    def account(self, n_bytes: int) -> bool:
        """Adds `n_bytes` to the dataset size and returns whether to write samples to disk."""
        self.bytes += n_bytes
        self.samples += 1
        if self.spilled:
            return True

        if self.bytes > self.budget_bytes:
            self._start_spilling(f"{self.bytes >> 20} MB of samples exceed the "
                                 f"{self.budget_bytes >> 20} MB budget")
            return True

        self._checks += 1
        if self._checks >= FREE_MEMORY_CHECK_INTERVAL:
            self._checks = 0
            available = available_memory()
            if available is not None and available < MIN_FREE_BYTES:
                self._start_spilling(f"only {available >> 20} MB RAM free")
                return True
        return False

    def _start_spilling(self, reason: str):
        self.spilled = True
        logger.info(f"Writing the dataset to disk: {reason} "
                    f"({self.samples} samples so far, target {self._target_dir()})")
        free = self._free_disk_bytes()
        if free is not None and free < self.bytes:
            logger.warning(f"Only {free >> 20} MB free in {self._target_dir()}, but the dataset is "
                           f"already {self.bytes >> 20} MB. Point OMMR4ALL_DATASET_CACHE_DIR at a "
                           f"larger filesystem.")

    def _target_dir(self) -> str:
        return self._dir or self.root or tempfile.gettempdir()

    def _free_disk_bytes(self) -> Optional[int]:
        try:
            return shutil.disk_usage(self._target_dir()).free
        except OSError:
            return None

    def path(self, suffix: str) -> str:
        """A fresh path inside the scratch directory, which is created on the first call."""
        if self._dir is None:
            if self.root:
                os.makedirs(self.root, exist_ok=True)
            self._dir = tempfile.mkdtemp(prefix='ommr4all-dataset-', dir=self.root)
            logger.info(f"Spilling dataset samples to {self._dir}")
        self._files += 1
        return os.path.join(self._dir, f"{self._files:08d}{suffix}")

    def close(self):
        if self._dir is not None:
            logger.info(f"Removing the dataset scratch directory {self._dir} "
                        f"({self._files} files, {self.bytes >> 20} MB of samples)")
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    def __enter__(self) -> 'DatasetSpill':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def dataset_spill(budget_bytes: Optional[int] = None) -> DatasetSpill:
    """The spill of one training run; use as a context manager around the whole training, the
    spilled samples must stay readable until the last epoch is done."""
    return DatasetSpill(budget_bytes)
