"""The memory budget of a training dataset (omr/dataset/spill.py).

Relevant in a container: psutil reports the free memory of the whole machine, so 40 % of it
can be above the container's memory limit and the training is OOM-killed instead of spilling
to disk. The budget therefore also looks at the memory cgroup.
"""
import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    stream=sys.stdout)

from omr.dataset import spill


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


class TestCgroupHeadroom(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='ommr4all-cgroup-')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_no_cgroup_files_means_unlimited(self):
        self.assertIsNone(spill.cgroup_headroom(self.root))

    def test_v2_limit_minus_usage(self):
        _write(os.path.join(self.root, 'memory.max'), '2000\n')
        _write(os.path.join(self.root, 'memory.current'), '500\n')
        self.assertEqual(1500, spill.cgroup_headroom(self.root))

    def test_v2_max_means_unlimited(self):
        _write(os.path.join(self.root, 'memory.max'), 'max\n')
        _write(os.path.join(self.root, 'memory.current'), '500\n')
        self.assertIsNone(spill.cgroup_headroom(self.root))

    def test_v2_reclaimable_page_cache_counts_as_free(self):
        # the trainers read every page image, so the usage is mostly page cache; without
        # this the headroom of a long run would collapse to nothing
        _write(os.path.join(self.root, 'memory.max'), '2000\n')
        _write(os.path.join(self.root, 'memory.current'), '1900\n')
        _write(os.path.join(self.root, 'memory.stat'), 'anon 100\ninactive_file 800\n')
        self.assertEqual(900, spill.cgroup_headroom(self.root))

    def test_a_cgroup_over_its_limit_has_no_headroom(self):
        _write(os.path.join(self.root, 'memory.max'), '2000\n')
        _write(os.path.join(self.root, 'memory.current'), '2500\n')
        self.assertEqual(0, spill.cgroup_headroom(self.root))

    def test_v1_limit_minus_usage(self):
        _write(os.path.join(self.root, 'memory', 'memory.limit_in_bytes'), '2000\n')
        _write(os.path.join(self.root, 'memory', 'memory.usage_in_bytes'), '600\n')
        _write(os.path.join(self.root, 'memory', 'memory.stat'), 'total_inactive_file 100\n')
        self.assertEqual(1500, spill.cgroup_headroom(self.root))

    def test_the_v1_unlimited_sentinel_is_not_a_limit(self):
        # cgroup v1 writes a huge number instead of a word
        _write(os.path.join(self.root, 'memory', 'memory.limit_in_bytes'), '9223372036854775807\n')
        _write(os.path.join(self.root, 'memory', 'memory.usage_in_bytes'), '600\n')
        self.assertIsNone(spill.cgroup_headroom(self.root))

    def test_unreadable_files_do_not_raise(self):
        _write(os.path.join(self.root, 'memory.max'), 'not-a-number\n')
        self.assertIsNone(spill.cgroup_headroom(self.root))


class TestAvailableMemory(unittest.TestCase):
    def test_the_cgroup_limit_wins_over_the_machine(self):
        with mock.patch.object(spill, 'cgroup_headroom', return_value=4 * 1024 ** 3):
            self.assertEqual(4 * 1024 ** 3, spill.available_memory())

    def test_the_machine_wins_when_it_is_the_tighter_one(self):
        with mock.patch.object(spill, 'cgroup_headroom', return_value=1 << 60):
            self.assertLess(spill.available_memory(), 1 << 60)

    def test_an_unlimited_cgroup_leaves_the_machine_value(self):
        with mock.patch.object(spill, 'cgroup_headroom', return_value=None):
            import psutil
            self.assertAlmostEqual(psutil.virtual_memory().available, spill.available_memory(),
                                   delta=512 * 1024 ** 2)

    def test_a_failing_cgroup_read_is_not_fatal(self):
        with mock.patch.object(spill, 'cgroup_headroom', side_effect=OSError('boom')):
            self.assertIsNotNone(spill.available_memory())

    def test_the_budget_follows_the_cgroup_limit(self):
        with mock.patch.object(spill, 'cgroup_headroom', return_value=10 * 1024 ** 3):
            self.assertEqual(int(10 * 1024 ** 3 * spill.DEFAULT_BUDGET_FRACTION),
                             spill.default_budget_bytes())

    def test_the_configured_budget_still_wins(self):
        with mock.patch.dict(os.environ, {'OMMR4ALL_DATASET_MEMORY_BUDGET_MB': '1024'}):
            self.assertEqual(1024 * 1024 ** 2, spill.default_budget_bytes())

    def test_the_budget_never_drops_below_the_floor(self):
        with mock.patch.object(spill, 'cgroup_headroom', return_value=64 * 1024 ** 2):
            self.assertEqual(spill.MIN_BUDGET_BYTES, spill.default_budget_bytes())


if __name__ == '__main__':
    unittest.main()
