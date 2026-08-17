import logging
import os
import sys
import unittest

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from omr.steps.text.guppy.progress import ProgressWriter, report_training_progress


class RecordingCallback:
    """Records what a TrainerCallback would receive."""

    def __init__(self):
        self.iterations = []
        self.best_models = []

    def next_iteration(self, iter, loss, acc):
        self.iterations.append((iter, loss, acc))

    def next_best_model(self, best_iter, best_acc, best_iters):
        self.best_models.append((best_iter, best_acc, best_iters))


class _Delegate:
    """Stands in for the real SummaryWriter; records that it still gets everything."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.scalars.append((tag, value, step))

    def add_text(self, *args, **kwargs):
        pass


def _run_epochs(writer: ProgressWriter, cers, batches_per_epoch=5):
    """Replay the scalar sequence guppyocr emits, one entry per epoch of `cers`."""
    for epoch, cer in enumerate(cers):
        for i in range(batches_per_epoch):
            writer.add_scalar('Loss/train', 1.0 / (epoch + 1), epoch * batches_per_epoch + i)
        last_step = epoch * batches_per_epoch + batches_per_epoch - 1
        writer.add_scalar('CER/val mean', cer, last_step)
        writer.add_scalar('CER/val total', cer, last_step)
        # emitted without a step by guppyocr, and last of the epoch
        writer.add_scalar('Line Accuracy', 1 - cer)


class TestProgressWriter(unittest.TestCase):
    def test_reports_epoch_progress(self):
        cb = RecordingCallback()
        delegate = _Delegate()
        writer = ProgressWriter(delegate, cb, n_epoch=4)

        _run_epochs(writer, [0.4, 0.3, 0.35, 0.33])

        # the epoch length is not visible from outside and has to be inferred
        self.assertEqual(writer._batches_per_epoch, 5)

        progress = [it / 4 for it, _, _ in cb.iterations]
        self.assertTrue(progress, 'no progress was reported')
        self.assertTrue(all(0 <= p <= 1 for p in progress), progress)
        self.assertTrue(all(b >= a for a, b in zip(progress, progress[1:])),
                        'progress must not move backwards: {}'.format(progress))
        self.assertEqual(progress[-1], 1.0)

        # everything is still forwarded to the real writer
        self.assertEqual(len(delegate.scalars), 4 * (5 + 3))

    def test_tracks_the_best_model_and_early_stopping_budget(self):
        cb = RecordingCallback()
        writer = ProgressWriter(_Delegate(), cb, n_epoch=4)

        # improves, improves, then two epochs without improvement
        _run_epochs(writer, [0.4, 0.3, 0.35, 0.33])

        self.assertEqual(len(cb.best_models), 4)
        best_iters = [b[0] for b in cb.best_models]
        without_improvement = [b[2] for b in cb.best_models]
        self.assertEqual(best_iters, [1, 2, 2, 2])
        self.assertEqual(without_improvement, [0, 0, 1, 2])
        # accuracy is 1 - the CER the best model was selected by
        self.assertAlmostEqual(cb.best_models[-1][1], 1 - 0.3, places=6)

    def test_a_failing_callback_cannot_abort_training(self):
        class Boom(RecordingCallback):
            def next_iteration(self, iter, loss, acc):
                raise RuntimeError('progress reporting is broken')

        delegate = _Delegate()
        writer = ProgressWriter(delegate, Boom(), n_epoch=2)
        _run_epochs(writer, [0.5, 0.4])       # must not raise
        self.assertEqual(len(delegate.scalars), 2 * (5 + 3))

    def test_unknown_attributes_are_forwarded(self):
        delegate = _Delegate()
        writer = ProgressWriter(delegate, RecordingCallback(), n_epoch=1)
        self.assertEqual(writer.add_text, delegate.add_text)


class TestReportTrainingProgress(unittest.TestCase):
    def test_without_a_callback_nothing_is_patched(self):
        try:
            from guppyocr import train_calamares
        except Exception as e:
            self.skipTest('guppyocr is not importable: {}'.format(e))

        original = train_calamares.SummaryWriter
        with report_training_progress(None, 10):
            self.assertIs(train_calamares.SummaryWriter, original)
        self.assertIs(train_calamares.SummaryWriter, original)

    def test_patches_and_restores_the_writer(self):
        try:
            from guppyocr import train_calamares
        except Exception as e:
            self.skipTest('guppyocr is not importable: {}'.format(e))

        original = train_calamares.SummaryWriter
        cb = RecordingCallback()
        try:
            with report_training_progress(cb, 10):
                self.assertIsNot(train_calamares.SummaryWriter, original)
                raise ValueError('training blew up')
        except ValueError:
            pass
        # restored even when the training raises
        self.assertIs(train_calamares.SummaryWriter, original)


if __name__ == '__main__':
    unittest.main()
