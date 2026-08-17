"""Host resource metrics (CPU/RAM/disk/GPU) for the administrative system
resources view.

Deliberately dependency-light and failure-tolerant: the view must render on a
CPU-only host and inside a Docker container started without GPU passthrough, so
every GPU query failure is reported as "unavailable" instead of raising.

Only relative load is reported. Absolute sizes, hardware models, driver and
library versions and the text of subprocess failures describe the host closely
enough to be useful when planning an attack, so they are written to the log
instead of to the API -- see _log_detail().
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


# nvidia-smi spawns a process; the client polls every few seconds and several
# admins may watch at once, so the parsed result is cached for a moment.
GPU_CACHE_TTL = 2.0

_gpu_cache_lock = threading.Lock()
_gpu_cache: Optional[Tuple[float, List[dict], bool]] = None

# the card model, the driver version, the temperature and the power draw are not queried at all:
# they say what this machine is rather than how busy it is, and the view must not disclose that
NVIDIA_SMI_QUERY = 'index,utilization.gpu,memory.used,memory.total'

# Probing torch has to happen in a separate process: the Django parent pins
# CUDA_VISIBLE_DEVICES='' so that no web worker grabs a GPU, and importing torch
# here would cost seconds and megabytes. The script reports what the wheel was
# built for, what the card is, and whether a kernel actually launches — the last
# one is the only thing that distinguishes a working setup from a torch build
# without kernels for this compute capability.
CUDA_PROBE = r'''
import json, sys
out = {}
try:
    import torch
except Exception as e:
    print(json.dumps({'error': 'torch could not be imported: %s' % e}))
    sys.exit(0)
out['torch_version'] = torch.__version__
out['cuda_built'] = torch.version.cuda
try:
    out['arch_list'] = list(torch.cuda.get_arch_list())
except Exception:
    out['arch_list'] = []
try:
    out['available'] = bool(torch.cuda.is_available())
except Exception as e:
    out['available'] = False
    out['error'] = str(e)[:500]
devices = []
if out.get('available'):
    for i in range(torch.cuda.device_count()):
        device = {'index': i}
        try:
            device['name'] = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
            device['capability'] = '%d.%d' % (major, minor)
            device['sm'] = 'sm_%d%d' % (major, minor)
            # None = unknown: some builds report an empty architecture list
            device['supported'] = device['sm'] in out['arch_list'] if out['arch_list'] else None
        except Exception as e:
            device['error'] = str(e)[:500]
        try:
            a = torch.rand(64, 64, device='cuda:%d' % i)
            float((a @ a).sum().item())
            torch.cuda.synchronize(i)
            device['compute_ok'] = True
        except Exception as e:
            device['compute_ok'] = False
            device['error'] = str(e)[:500]
        devices.append(device)
out['devices'] = devices
print(json.dumps(out))
'''

CUDA_PROBE_TIMEOUT = 180

_cuda_lock = threading.Lock()
_cuda_result = None            # None => never run
_cuda_running = False


def _log_detail(what: str, detail) -> None:
    """Keep a diagnostic detail out of the API but available to whoever runs the server."""
    logger.warning('%s: %s', what, detail)


def _percent(used, total):
    if not used or not total:
        return 0.0 if total else None
    return round(used * 100 / total, 1)


def _num(value: str, cast):
    """nvidia-smi prints '[N/A]' or '[Not Supported]' for unavailable fields."""
    value = value.strip()
    if not value or value.startswith('['):
        return None
    try:
        return cast(value)
    except ValueError:
        return None


def _nvidia_smi_env() -> dict:
    # the Django parent process pins CUDA_VISIBLE_DEVICES='' (manage.py, wsgi.py)
    # so that no web worker ever grabs a GPU; nvidia-smi must not inherit it
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def _query_gpus() -> Tuple[List[dict], bool]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=' + NVIDIA_SMI_QUERY, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, env=_nvidia_smi_env(),
        )
    except FileNotFoundError:
        logger.debug('nvidia-smi was not found on this host')
        return [], False
    except subprocess.TimeoutExpired:
        _log_detail('nvidia-smi did not respond within 3 s', '')
        return [], False
    except OSError as e:
        _log_detail('nvidia-smi could not be started', e)
        return [], False

    if result.returncode != 0:
        _log_detail('nvidia-smi exited with code {}'.format(result.returncode),
                    (result.stderr or result.stdout or '').strip())
        return [], False

    gpus = []
    for line in result.stdout.strip().splitlines():
        fields = line.split(',')
        if len(fields) < 4:
            logger.warning('Unexpected nvidia-smi output line: %s', line)
            continue
        gpus.append({
            'index': _num(fields[0], int),
            'utilization': _num(fields[1], float),
            # the absolute amount of graphics memory is the model of the card by another name
            'memory_percent': _percent(_num(fields[2], float), _num(fields[3], float)),
        })
    return gpus, True


def gpu_stats(use_cache: bool = True) -> Tuple[List[dict], bool]:
    """Per-GPU load, and whether the GPU metrics could be obtained at all."""
    global _gpu_cache
    with _gpu_cache_lock:
        if use_cache and _gpu_cache is not None and time.time() - _gpu_cache[0] < GPU_CACHE_TTL:
            return _gpu_cache[1], _gpu_cache[2]

        gpus, available = _query_gpus()
        _gpu_cache = (time.time(), gpus, available)
        return gpus, available


def _run_cuda_probe() -> dict:
    try:
        result = subprocess.run(
            [sys.executable, '-c', CUDA_PROBE],
            capture_output=True, text=True, timeout=CUDA_PROBE_TIMEOUT, env=_nvidia_smi_env(),
        )
    except subprocess.TimeoutExpired:
        return {'error': 'the CUDA check did not finish within {} s'.format(CUDA_PROBE_TIMEOUT)}
    except OSError as e:
        return {'error': 'the CUDA check could not be started: {}'.format(e)}

    if result.returncode != 0:
        message = (result.stderr or '').strip().splitlines()
        return {'error': message[-1] if message else
                'the CUDA check exited with code {}'.format(result.returncode)}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning('Unexpected CUDA probe output: %s', result.stdout[:500])
        return {'error': 'the CUDA check returned unreadable output'}


def _reduced_cuda(result: dict) -> dict:
    """The probe result without the torch/CUDA versions, device models and error text.

    Those name the exact software stack of the server; all the view needs to know is whether torch
    can compute on each device. The full result is logged once by _log_cuda_result().
    """
    return {
        'available': bool(result.get('available')),
        'devices': [{'index': device.get('index'), 'compute_ok': bool(device.get('compute_ok'))}
                    for device in result.get('devices', [])],
    }


def _log_cuda_result(result: dict) -> None:
    """Log what the API no longer reports, once per probe run rather than once per request."""
    if result.get('error'):
        _log_detail('CUDA check failed', result['error'])
    for device in result.get('devices', []):
        if device.get('error'):
            _log_detail('CUDA device {} is not usable'.format(device.get('index')), device['error'])
    logger.info('CUDA check: torch %s built for CUDA %s, architectures %s, devices %s',
                result.get('torch_version'), result.get('cuda_built'), result.get('arch_list'),
                [{k: v for k, v in d.items() if k != 'error'} for d in result.get('devices', [])])


def cuda_status(refresh: bool = False) -> dict:
    """Whether torch can actually compute on the GPUs of this host.

    The probe imports torch in a subprocess, which takes seconds, so it runs in
    the background and the caller gets ``{'state': 'checking'}`` until it is
    done. The result cannot change without restarting the server, so it is kept
    until explicitly refreshed.
    """
    global _cuda_result, _cuda_running

    with _cuda_lock:
        if _cuda_result is not None and not refresh:
            return dict(_reduced_cuda(_cuda_result), state='ready')
        if _cuda_running:
            return {'state': 'checking'}
        _cuda_running = True

    def probe():
        global _cuda_result, _cuda_running
        result = _run_cuda_probe()
        _log_cuda_result(result)
        with _cuda_lock:
            _cuda_result = result
            _cuda_running = False

    threading.Thread(target=probe, name='cuda_probe', daemon=True).start()
    return {'state': 'checking'}


def _disk(label: str, path: str) -> Optional[dict]:
    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        logger.debug('Could not stat %s: %s', path, e)
        return None
    # neither the absolute path nor the size of the volume is reported: that is server-side
    # layout which the administrative view does not need in order to show how full it is
    return {
        'label': label,
        'percent': _percent(usage.used, usage.total),
    }


def disk_usage() -> List[dict]:
    """Free space of the two directories the server actually writes to."""
    from django.conf import settings

    paths = [('storage', settings.PRIVATE_MEDIA_ROOT),
             ('database', os.path.dirname(settings.DATABASES['default']['NAME']))]
    disks = []
    seen = set()
    for label, path in paths:
        if not path:
            continue
        # storage and db commonly live on the same volume; report it once
        try:
            key = os.stat(path).st_dev
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        disk = _disk(label, path)
        if disk:
            disks.append(disk)
    return disks


def process_state(pid: Optional[int]) -> Optional[str]:
    """The process state as psutil reports it ('running', 'sleeping', 'disk-sleep', ...), or None.

    'disk-sleep' is uninterruptible sleep (D): the process is blocked in the kernel and
    accepts neither SIGTERM nor SIGKILL. That is the state a worker ends up in when its
    GPU driver or a mount wedges, and it is worth showing to an administrator, because
    it is the one case where a slot cannot be cleaned up by signalling alone.
    """
    if pid is None:
        return None
    try:
        return psutil.Process(pid).status()
    except Exception:
        return None


def cpu_and_memory() -> dict:
    """CPU load and memory usage of the host, as a percentage of what it has.

    ``cpu_percent`` is sampled without blocking, i.e. it reports the average
    since the previous call. The first call after server start therefore
    returns 0 — acceptable for a view that polls continuously.

    Core counts, the load average and the installed amount of memory are how much machine this is,
    not how busy it currently is, and are therefore not reported.
    """
    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
    except (OSError, RuntimeError):       # e.g. containers without /proc/vmstat
        swap = None

    return {
        'cpu': {
            'percent': psutil.cpu_percent(interval=None),
        },
        'memory': {
            'percent': memory.percent,
        },
        'swap': {
            'percent': swap.percent,
        } if swap and swap.total else None,
    }
