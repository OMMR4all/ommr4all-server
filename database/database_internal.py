import os
import ommr4all.settings as settings

INTERNAL_STORAGE = settings.INTERNAL_STORAGE_ROOT

# Not INTERNAL_STORAGE/default_models: the default models are the writable part of the
# internal storage and can live on a volume (OMMR4ALL_DEFAULT_MODELS_ROOT, see settings.py).
DEFAULT_MODELS = settings.DEFAULT_MODELS_ROOT
DEFAULT_VIRTUAL_KEYBOARDS = os.path.join(INTERNAL_STORAGE, 'default_virtual_keyboards')
