import multiprocessing
import os
import sys

# Keep CUDA out of the parent (Django/Apache) process; the spawned task
# workers set CUDA_VISIBLE_DEVICES to their assigned GPU themselves
# (see restapi/operationworker/taskworkermain.py).
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

# Task workers are spawned via sys.executable, which under mod_wsgi may point
# at the Apache binary instead of this venv's interpreter.
_python = os.path.join(sys.exec_prefix, 'bin', 'python3')
if os.path.exists(_python):
    multiprocessing.set_executable(_python)

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')

application = get_wsgi_application()
