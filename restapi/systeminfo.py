"""Host resource metrics (CPU/RAM/disk/GPU) for the administrative system
resources view.

Deliberately dependency-light and failure-tolerant: the view must render on a
CPU-only host and inside a Docker container started without GPU passthrough, so
every GPU query failure is reported as a message instead of raising.
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
_gpu_cache: Optional[Tuple[float, List[dict], Optional[str]]] = None

NVIDIA_SMI_QUERY = ('index,name,driver_version,utilization.gpu,memory.used,memory.total,'
                    'temperature.gpu,power.draw,power.limit')

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


def _query_gpus() -> Tuple[List[dict], Optional[str]]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=' + NVIDIA_SMI_QUERY, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, env=_nvidia_smi_env(),
        )
    except FileNotFoundError:
        return [], 'nvidia-smi was not found on this host'
    except subprocess.TimeoutExpired:
        return [], 'nvidia-smi did not respond within 3 s'
    except OSError as e:
        return [], 'nvidia-smi could not be started: {}'.format(e)

    if result.returncode != 0:
        message = (result.stderr or result.stdout or '').strip().splitlines()
        return [], message[0] if message else 'nvidia-smi exited with code {}'.format(result.returncode)

    gpus = []
    for line in result.stdout.strip().splitlines():
        fields = line.split(',')
        if len(fields) < 9:
            logger.warning('Unexpected nvidia-smi output line: %s', line)
            continue
        gpus.append({
            'index': _num(fields[0], int),
            'name': fields[1].strip(),
            'driver_version': fields[2].strip(),
            'utilization': _num(fields[3], float),
            'memory_used': _num(fields[4], float),      # MiB
            'memory_total': _num(fields[5], float),     # MiB
            'temperature': _num(fields[6], float),
            'power_draw': _num(fields[7], float),
            'power_limit': _num(fields[8], float),
        })
    return gpus, None


def gpu_stats(use_cache: bool = True) -> Tuple[List[dict], Optional[str]]:
    """Per-GPU hardware metrics, or ([], reason) if they cannot be obtained."""
    global _gpu_cache
    with _gpu_cache_lock:
        if use_cache and _gpu_cache is not None and time.time() - _gpu_cache[0] < GPU_CACHE_TTL:
            return _gpu_cache[1], _gpu_cache[2]

        gpus, error = _query_gpus()
        _gpu_cache = (time.time(), gpus, error)
        return gpus, error


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
            return dict(_cuda_result, state='ready')
        if _cuda_running:
            return {'state': 'checking'}
        _cuda_running = True

    def probe():
        global _cuda_result, _cuda_running
        result = _run_cuda_probe()
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
    # the absolute path is deliberately not reported: it is server-side layout
    # that the administrative view does not need
    return {
        'label': label,
        'used': usage.used,
        'total': usage.total,
        'free': usage.free,
        'percent': round(usage.used * 100 / usage.total, 1) if usage.total else 0,
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


def cpu_and_memory() -> dict:
    """CPU load and memory usage of the host.

    ``cpu_percent`` is sampled without blocking, i.e. it reports the average
    since the previous call. The first call after server start therefore
    returns 0 — acceptable for a view that polls continuously.
    """
    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
    except (OSError, RuntimeError):       # e.g. containers without /proc/vmstat
        swap = None
    try:
        load_avg = [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):     # not available on all platforms
        load_avg = None

    return {
        'cpu': {
            'percent': psutil.cpu_percent(interval=None),
            'per_cpu': psutil.cpu_percent(interval=None, percpu=True),
            'count': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'load_avg': load_avg,
        },
        'memory': {
            'used': memory.total - memory.available,
            'total': memory.total,
            'percent': memory.percent,
        },
        'swap': {
            'used': swap.used,
            'total': swap.total,
            'percent': swap.percent,
        } if swap else None,
    }
