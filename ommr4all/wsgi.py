import os

# Prevent CUDA from initializing in the parent (Django/Apache) process.
# If any third-party import transitively pulls in torch before a fork(),
# CUDA must not be active or the forked child process will see corrupt state.
# Worker children override this via CUDA_VISIBLE_DEVICES in taskworkerthread.py.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')

application = get_wsgi_application()
