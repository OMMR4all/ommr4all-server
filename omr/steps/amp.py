"""Which dtype automatic mixed precision uses on this GPU.

`torch.cuda.is_bf16_supported()` is not the question one usually means: without arguments it
also counts *emulated* bfloat16, so it returns True on Turing (sm_75) and Pascal (sm_61),
where there is no hardware for it. Autocasting to bf16 there costs the tensor cores and runs
far slower than fp16 while buying nothing. Hardware bf16 starts at sm_80 (Ampere).

fp16 has the smaller exponent range, which is why training needs a loss scaler (the callers
already create one for exactly that case). Overflow-sensitive parts of the network - the
pointer head and its soft-argmax - force fp32 themselves, see omr/steps/end2end/networks.py.

OMMR4ALL_AMP_DTYPE overrides the choice: 'auto' (default), 'bf16', 'fp16' or 'fp32'
('fp32' disables autocast altogether, i.e. the pre-mixed-precision behaviour).
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

AMP_DTYPE_ENV = 'OMMR4ALL_AMP_DTYPE'
# compute capability from which bfloat16 exists in hardware (Ampere)
BF16_MIN_MAJOR = 8


def _configured() -> Optional[str]:
    raw = (os.environ.get(AMP_DTYPE_ENV) or '').strip().lower()
    if not raw or raw == 'auto':
        return None
    if raw in ('bf16', 'bfloat16'):
        return 'bf16'
    if raw in ('fp16', 'float16', 'half'):
        return 'fp16'
    if raw in ('fp32', 'float32', 'none', 'off'):
        return 'fp32'
    logger.warning(f"Ignoring {AMP_DTYPE_ENV}={raw!r}; use auto, bf16, fp16 or fp32")
    return None


def has_hardware_bf16() -> bool:
    """Whether the current CUDA device computes bfloat16 natively rather than emulating it."""
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        # the keyword exists since torch 2.6; without it the function reports emulation too
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).major >= BF16_MIN_MAJOR


def autocast_dtype(device) -> Optional['object']:
    """The dtype to autocast to on `device`, or None for no autocast (CPU, or fp32 requested)."""
    import torch
    if getattr(device, 'type', device) != 'cuda':
        return None

    choice = _configured()
    if choice == 'fp32':
        return None
    if choice is None:
        choice = 'bf16' if has_hardware_bf16() else 'fp16'
    return torch.bfloat16 if choice == 'bf16' else torch.float16
