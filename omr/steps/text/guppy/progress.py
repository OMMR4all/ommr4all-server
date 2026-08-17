"""Training progress for the guppy OCR trainer.

``guppyocr.train_calamares.train_model`` takes no callback -- the package has no
progress hook at all. It does however report every epoch through a TensorBoard
``SummaryWriter`` that it creates itself, and that writer is a plain module level
name, so substituting it is the one interception point that needs no change to the
library:

    train_calamares.py:25   from torch.utils.tensorboard import SummaryWriter
    train_calamares.py:284  writer = SummaryWriter()

The proxy below wraps a real writer (so the TensorBoard output is unchanged) and
translates the scalars guppyocr already emits into ``TrainerCallback`` calls:

    'Loss/train'       per batch, global_step = epoch * num_batch + i
    'CER/val mean'     once per epoch
    'CER/val total'    once per epoch  -- the metric the best model is chosen by
    'Line Accuracy'    once per epoch, last of the three and without a step

``num_batch`` is not visible from outside, so it is derived at the first epoch
boundary: 'CER/val total' carries ``epoch * num_batch + (num_batch - 1)``.
"""
import logging
from contextlib import contextmanager
from typing import Optional

from omr.steps.algorithm import TrainerCallback

logger = logging.getLogger(__name__)

TAG_LOSS = 'Loss/train'
TAG_VAL_CER = 'CER/val total'
TAG_LINE_ACCURACY = 'Line Accuracy'


class ProgressWriter:
    """SummaryWriter stand-in that reports guppyocr's scalars to a TrainerCallback.

    Everything except ``add_scalar`` is forwarded to a real writer, so TensorBoard
    keeps working exactly as before.
    """

    def __init__(self, delegate, callback: TrainerCallback, n_epoch: int):
        self._delegate = delegate
        self._callback = callback
        self._n_epoch = max(int(n_epoch), 1)

        self._epoch = 0                     # epochs completed so far
        self._batches_per_epoch: Optional[int] = None
        self._last_loss = -1.0
        self._last_accuracy = -1.0
        self._best_cer: Optional[float] = None
        self._best_epoch = 0
        self._epochs_without_improvement = 0

    def __getattr__(self, item):
        # only reached for attributes this class does not define
        return getattr(self._delegate, item)

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        try:
            self._observe(tag, scalar_value, global_step)
        except Exception as e:
            # progress reporting must never be able to abort a training run
            logger.warning('Could not report guppy training progress: %s', e)
        return self._delegate.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def _observe(self, tag, value, global_step):
        value = float(value)
        if tag == TAG_LOSS:
            self._last_loss = value
            self._report_batch(global_step)
        elif tag == TAG_VAL_CER:
            if self._batches_per_epoch is None and global_step is not None:
                # first epoch boundary: global_step == epoch * num_batch + (num_batch - 1)
                self._batches_per_epoch = max(int(global_step) + 1, 1)
            self._track_best(value)
        elif tag == TAG_LINE_ACCURACY:
            # last scalar of the epoch, so the epoch is complete here
            self._last_accuracy = value
            self._complete_epoch()

    def _report_batch(self, global_step):
        """Smooth within-epoch progress, once the epoch length is known."""
        if self._batches_per_epoch is None or global_step is None:
            return
        batch = int(global_step) - self._epoch * self._batches_per_epoch
        if batch < 0:
            return
        fraction = min(batch / self._batches_per_epoch, 1.0)
        # a float iteration is fine: the task runner only computes iter / total_iters
        self._callback.next_iteration(self._epoch + fraction, self._last_loss, self._last_accuracy)

    def _track_best(self, total_cer: float):
        """guppyocr keeps the model with the lowest total CER but reports no signal
        for it (train_calamares.py:585-597), so mirror its bookkeeping."""
        if self._best_cer is None or total_cer < self._best_cer:
            self._best_cer = total_cer
            self._best_epoch = self._epoch + 1
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

    def _complete_epoch(self):
        self._epoch += 1
        self._callback.next_iteration(min(self._epoch, self._n_epoch), self._last_loss, self._last_accuracy)
        if self._best_cer is not None:
            self._callback.next_best_model(self._best_epoch, 1.0 - self._best_cer,
                                           self._epochs_without_improvement)


@contextmanager
def report_training_progress(callback: Optional[TrainerCallback], n_epoch: int):
    """Report guppyocr's training progress to ``callback`` for the duration of the block.

    A no-op without a callback, and the patched attribute is always restored.
    """
    if callback is None:
        yield
        return

    from guppyocr import train_calamares

    original = train_calamares.SummaryWriter

    def factory(*args, **kwargs):
        return ProgressWriter(original(*args, **kwargs), callback, n_epoch)

    train_calamares.SummaryWriter = factory
    try:
        yield
    finally:
        train_calamares.SummaryWriter = original
