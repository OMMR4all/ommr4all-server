"""Where pretrained base weights downloaded at training time are kept.

Most pretrained weights are fetched by libraries that have a cache of their own and honour an
environment variable for it: the Hugging Face hub (Swin, Chandra, DINO) uses ``HF_HOME``,
torchvision (ConvNeXt, Mask R-CNN) uses ``TORCH_HOME``. The deployment points both at a
volume, so the downloads survive a restart.

ultralytics is the exception. ``YOLO("yolo11n.pt")`` resolves the name against
``SETTINGS['weights_dir']``, which is the *relative* path ``weights``, and downloads the
checkpoint into the current working directory when it is not there. Under mod_wsgi the
working directory is ``/``, and in a container it is thrown away on the next start. Passing
an absolute path inside ultralytics' own config directory (``YOLO_CONFIG_DIR``, which the
deployment puts on the same volume) makes the download land in the cache instead.
"""
import logging
import os

logger = logging.getLogger(__name__)


def yolo_base_weights(name: str) -> str:
    """Absolute path of a YOLO base checkpoint, downloaded there on first use.

    Falls back to the plain name if the directory cannot be created, which restores the
    old behaviour (download into the working directory) rather than failing a training.
    """
    try:
        from ultralytics.utils import USER_CONFIG_DIR
        directory = os.path.join(str(USER_CONFIG_DIR), 'weights')
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, name)
    except Exception as e:
        logger.warning('Could not provide a cache directory for {}, downloading it into the '
                       'working directory instead: {}'.format(name, e))
        return name
