"""Which dtype autocast uses (omr/steps/amp.py).

torch.cuda.is_bf16_supported() counts emulated bfloat16, so it is True on Turing and Pascal
where there is no hardware for it; autocasting to bf16 there loses the tensor cores. These
tests pin the decision and the OMMR4ALL_AMP_DTYPE escape hatch, without needing a GPU.
"""
import logging
import os
import sys
import unittest
from unittest import mock

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    stream=sys.stdout)

import torch

from omr.steps import amp


class TestAutocastDtype(unittest.TestCase):
    def setUp(self):
        # a clean environment; individual tests patch it where they need to
        self.env = mock.patch.dict(os.environ, {amp.AMP_DTYPE_ENV: ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_cpu_never_autocasts(self):
        self.assertIsNone(amp.autocast_dtype(torch.device('cpu')))

    def test_hardware_bf16_gets_bf16(self):
        with mock.patch.object(amp, 'has_hardware_bf16', return_value=True):
            self.assertEqual(torch.bfloat16, amp.autocast_dtype(torch.device('cuda')))

    def test_without_hardware_bf16_fp16_is_used(self):
        # Turing/Pascal: emulated bf16 would run without tensor cores
        with mock.patch.object(amp, 'has_hardware_bf16', return_value=False):
            self.assertEqual(torch.float16, amp.autocast_dtype(torch.device('cuda')))

    def test_the_environment_can_force_bf16(self):
        with mock.patch.dict(os.environ, {amp.AMP_DTYPE_ENV: 'bf16'}), \
                mock.patch.object(amp, 'has_hardware_bf16', return_value=False):
            self.assertEqual(torch.bfloat16, amp.autocast_dtype(torch.device('cuda')))

    def test_the_environment_can_force_fp16(self):
        with mock.patch.dict(os.environ, {amp.AMP_DTYPE_ENV: 'float16'}), \
                mock.patch.object(amp, 'has_hardware_bf16', return_value=True):
            self.assertEqual(torch.float16, amp.autocast_dtype(torch.device('cuda')))

    def test_fp32_disables_autocast(self):
        for value in ('fp32', 'float32', 'off'):
            with mock.patch.dict(os.environ, {amp.AMP_DTYPE_ENV: value}), \
                    mock.patch.object(amp, 'has_hardware_bf16', return_value=False):
                self.assertIsNone(amp.autocast_dtype(torch.device('cuda')), value)

    def test_an_unknown_value_falls_back_to_the_automatic_choice(self):
        with mock.patch.dict(os.environ, {amp.AMP_DTYPE_ENV: 'int8'}), \
                mock.patch.object(amp, 'has_hardware_bf16', return_value=True):
            self.assertEqual(torch.bfloat16, amp.autocast_dtype(torch.device('cuda')))

    def test_a_device_without_a_type_attribute_is_accepted(self):
        with mock.patch.object(amp, 'has_hardware_bf16', return_value=True):
            self.assertEqual(torch.bfloat16, amp.autocast_dtype('cuda'))
        self.assertIsNone(amp.autocast_dtype('cpu'))


class TestHardwareBf16(unittest.TestCase):
    def test_no_cuda_means_no_bf16(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertFalse(amp.has_hardware_bf16())

    def test_the_emulation_flag_is_passed(self):
        probe = mock.Mock(return_value=False)
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'is_bf16_supported', probe):
            self.assertFalse(amp.has_hardware_bf16())
        probe.assert_called_once_with(including_emulation=False)

    def test_an_older_torch_without_the_flag_uses_the_compute_capability(self):
        props = mock.Mock(major=7)   # Turing
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'is_bf16_supported', side_effect=TypeError), \
                mock.patch.object(torch.cuda, 'current_device', return_value=0), \
                mock.patch.object(torch.cuda, 'get_device_properties', return_value=props):
            self.assertFalse(amp.has_hardware_bf16())
            props.major = 8          # Ampere
            self.assertTrue(amp.has_hardware_bf16())


if __name__ == '__main__':
    unittest.main()
