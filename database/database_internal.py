import os
from django.conf import settings

INTERNAL_STORAGE = os.path.join(settings.BASE_DIR, 'internal_storage')

DEFAULT_MODELS = os.path.join(INTERNAL_STORAGE, 'default_models')
DEFAULT_VIRTUAL_KEYBOARDS = os.path.join(INTERNAL_STORAGE, 'default_virtual_keyboards')
