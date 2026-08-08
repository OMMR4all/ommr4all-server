from .meta import ModelMeta, ModelTrainingInfo, ModelUsage
from .definitions import MetaId
from typing import Optional
import os
import logging
import datetime
import shutil

logger = logging.getLogger(__name__)


class Model:
    META_FILE = 'meta.json'
    USAGE_FILE = 'usage.json'
    TRAINING_FILE = 'training.json'

    # bookkeeping a model directory holds besides its weights; a directory with nothing else
    # never produced a model (a training run that failed, or a placeholder directory)
    NON_WEIGHT_FILES = {META_FILE, USAGE_FILE, TRAINING_FILE, 'dataset_params.json'}

    @staticmethod
    def from_id_str(id: str, meta: Optional[ModelMeta] = None) -> Optional['Model']:
        try:
            return Model(MetaId.from_str(id), meta)
        except Exception as e:
            logger.error('Error parsing id {}'.format(id))
            logger.exception(e)
            return None

    def __init__(self, meta_id: MetaId, meta: Optional[ModelMeta] = None):
        self.meta_id = meta_id
        self.path = meta_id.path()
        self.meta_path = os.path.join(self.path, Model.META_FILE)
        self.name = meta_id.name

        self._meta: Optional[ModelMeta] = meta

    def id(self) -> str:
        return str(self.meta_id)

    def meta(self) -> ModelMeta:
        if not self._meta:
            try:
                with open(self.meta_path, 'r') as f:
                    self._meta = ModelMeta.from_json(f.read())
            except (FileNotFoundError, ValueError):
                logger.warning("ModelMeta file not existing at {}. Creating a new one!".format(self.meta_path))
                self._meta = ModelMeta(
                    id=self.id(),
                    created=datetime.datetime.now(),
                )

        self._meta.id = self.id()
        return self._meta

    def save_meta(self):
        os.makedirs(self.path, exist_ok=True)
        if self._meta:
            self._meta.id = self.id()
            with open(self.meta_path, 'w') as f:
                f.write(self._meta.to_json())

    def usage(self) -> ModelUsage:
        """When this model was last used for a prediction; all-zero when it never was."""
        try:
            with open(self.local_file(Model.USAGE_FILE), 'r') as f:
                return ModelUsage.from_json(f.read())
        except (FileNotFoundError, ValueError):
            return ModelUsage()

    def mark_used(self):
        """Record a prediction with this model. Never raises: usage data is not worth a failed task.

        Written to its own file rather than into meta.json, whose mtime is part of the predictor
        cache key (see omr/steps/predictorcache.py).
        """
        try:
            usage = self.usage()
            usage.last_used = datetime.datetime.now()
            usage.n_used += 1
            tmp = self.local_file(Model.USAGE_FILE + '.tmp')
            with open(tmp, 'w') as f:
                f.write(usage.to_json())
            os.replace(tmp, self.local_file(Model.USAGE_FILE))
        except Exception as e:
            logger.warning('Could not record the usage of model {}'.format(self.path))
            logger.exception(e)

    def training_info(self) -> Optional[ModelTrainingInfo]:
        """What this model was trained on, or None for models trained before this was recorded."""
        try:
            with open(self.local_file(Model.TRAINING_FILE), 'r') as f:
                return ModelTrainingInfo.from_json(f.read())
        except (FileNotFoundError, ValueError):
            return None

    def save_training_info(self, info: ModelTrainingInfo):
        """Record the training run. Never raises: provenance is not worth a failed training."""
        try:
            os.makedirs(self.path, exist_ok=True)
            tmp = self.local_file(Model.TRAINING_FILE + '.tmp')
            with open(tmp, 'w') as f:
                f.write(info.to_json())
            os.replace(tmp, self.local_file(Model.TRAINING_FILE))
        except Exception as e:
            logger.warning('Could not store the training info of model {}'.format(self.path))
            logger.exception(e)

    def has_weights(self) -> bool:
        """Whether the directory holds anything a predictor could load.

        ``exists()`` only tells that a meta file is there, which is also true for the placeholder
        directories of steps that never shipped a default model.
        """
        for root, _, files in os.walk(self.path):
            for name in files:
                if root != self.path or name not in Model.NON_WEIGHT_FILES:
                    return True
        return False

    def size(self) -> int:
        """Bytes occupied by the model directory."""
        total = 0
        for root, _, files in os.walk(self.path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total

    def local_file(self, file: str) -> str:
        return os.path.join(self.path, file)

    def exists(self, file: str = None) -> bool:
        return os.path.exists(self.local_file(file if file is not None else Model.META_FILE))

    def delete(self):
        if self.exists(''):
            shutil.rmtree(self.path)

    def copy_to(self, target_model: 'Model', override=True):
        if not self.exists():
            raise FileNotFoundError()

        if not override and target_model.exists():
            raise FileExistsError()

        copyied_model = Model(target_model.meta_id, meta=self.meta())
        # save_meta rewrites the id to the target, so remember where the copy came from --
        # that is what marks the source model as backing a default (see restapi/views/administrativemodels.py)
        copyied_model._meta.source_id = self.id()
        self._meta = None
        shutil.rmtree(target_model.path, ignore_errors=True)
        shutil.copytree(self.path, target_model.path)
        copyied_model.save_meta()

