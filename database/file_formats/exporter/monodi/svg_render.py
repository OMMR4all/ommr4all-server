"""Rendering a monodi document into SVG.

`internal_storage/resources/monodi_svg_render` is a bundled JVM tool (jars plus the
`bin/one-shot` launcher), so rendering needs a Java runtime *on the machine*. Without one the
launcher prints "No java installations was detected" on **stdout** and exits 1. The callers
used to pull the `<svg>` element out of that output with a regex and dereference the match
directly, so a missing JVM surfaced as `AttributeError: 'NoneType' object has no attribute
'group'` - a 500 with a traceback that says nothing about the actual cause. The Docker image
shipped without a JRE, which is exactly that case.

The helpers here keep the one deliberate tolerance of the old code: the exit code of the
launcher is *not* trusted. Output that contains an SVG counts as success even when the tool
exits non-zero (it does so on recoverable layout complaints); only output without an SVG is
an error.
"""
import logging
import os
import re
import shutil
import subprocess
from typing import List, Optional

import ommr4all.settings as settings

logger = logging.getLogger(__name__)

SCRIPT_PATH = os.path.join(settings.INTERNAL_STORAGE_ROOT, 'resources', 'monodi_svg_render',
                           'bin', 'one-shot')
# starting a JVM plus laying out one document; seconds in practice. Without a limit a hung
# renderer would hold an Apache worker thread forever.
RENDER_TIMEOUT_SECONDS = 120

_SVG_RE = re.compile(r'(<svg.*</svg>)', re.DOTALL)
# what the launcher prints when it cannot find a JVM (bin/one-shot, java_version_check)
_NO_JAVA = 'No java installations was detected'


class MonodiSvgRenderError(RuntimeError):
    """The renderer produced no SVG. The message names the reason as far as it is known."""


def java_available() -> bool:
    """Whether the launcher will find a JVM: JAVA_HOME first, then the PATH, as it does."""
    java_home = (os.environ.get('JAVA_HOME') or '').strip()
    if java_home and os.access(os.path.join(java_home, 'bin', 'java'), os.X_OK):
        return True
    return shutil.which('java') is not None


def available() -> bool:
    """Whether rendering can work at all - the launcher is there and it will find a JVM."""
    return os.access(SCRIPT_PATH, os.X_OK) and java_available()


def render(monodi_json: str, width) -> str:
    """The SVG of a monodi document (its json as a string), rendered at `width` pixels.

    Raises MonodiSvgRenderError if no SVG came out of the renderer.
    """
    return _extract(_run(['-', '-w', str(width)], stdin=monodi_json))


def render_file(json_path: str, svg_path: str) -> None:
    """Renders the monodi json at `json_path` into `svg_path` (what the renderer writes)."""
    _run([json_path, '-o', svg_path])
    if not os.path.exists(svg_path):
        raise MonodiSvgRenderError('the renderer wrote no file to {}'.format(svg_path))


def _run(args: List[str], stdin: Optional[str] = None) -> str:
    if not os.path.exists(SCRIPT_PATH):
        raise MonodiSvgRenderError(
            'the renderer is missing at {}'.format(SCRIPT_PATH))
    try:
        proc = subprocess.run([SCRIPT_PATH] + args, input=stdin,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=RENDER_TIMEOUT_SECONDS)
    except FileNotFoundError as e:
        raise MonodiSvgRenderError('could not start the renderer: {}'.format(e))
    except subprocess.TimeoutExpired:
        raise MonodiSvgRenderError(
            'the renderer did not finish within {} s'.format(RENDER_TIMEOUT_SECONDS))

    if proc.returncode != 0:
        # not fatal by itself, see the module docstring
        logger.warning('The monodi SVG renderer exited with %s: %s',
                       proc.returncode, _first_lines(proc.stdout, proc.stderr))
    return proc.stdout or ''


def _extract(output: str) -> str:
    match = _SVG_RE.search(output)
    if match:
        return match.group(1)

    if _NO_JAVA in output or not java_available():
        raise MonodiSvgRenderError(
            'no Java runtime: the renderer is a JVM tool and needs one on the machine '
            '(Debian/Ubuntu: openjdk-17-jre-headless). JAVA_HOME={!r}'.format(
                os.environ.get('JAVA_HOME', '')))
    raise MonodiSvgRenderError(
        'the renderer produced no SVG: {}'.format(_first_lines(output, '')))


def _first_lines(stdout: str, stderr: str, limit: int = 400) -> str:
    text = ' | '.join(part.strip() for part in (stdout, stderr) if part and part.strip())
    return text[:limit] if text else '<no output>'
