from .meta import ModelMeta
from .definitions import MetaId
from typing import Optional
import os
import logging
import datetime
import shutil

logger = logging.getLogger(__name__)


class Model:
    META_FILE = 'meta.json'

    @staticmethod
    def from_id_str(id: str, meta: Optional[ModelMeta] = None) -> Optional['Model']:
        try:
            return Model(MetaId.from_str(id), meta)
        except Exception as e:
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
            except FileNotFoundError:
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
                f.write(self._meta.to_json(indent=2))

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
        self._meta = None
        shutil.rmtree(target_model.path)
        shutil.copytree(self.path, target_model.path)
        copyied_model.save_meta()

