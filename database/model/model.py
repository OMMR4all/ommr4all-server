from .meta import ModelMeta
from typing import Optional
import os
import logging
import uuid
import datetime
import shutil
from ommr4all.settings import BASE_DIR

logger = logging.getLogger(__name__)


class Model:
    META_FILE = 'meta.json'

    @staticmethod
    def from_id(id: str, meta: Optional[ModelMeta] = None):
        return Model(os.path.join(BASE_DIR, id), meta)

    def __init__(self, path: str, meta: Optional[ModelMeta] = None):
        self.path = path
        self.meta_path = os.path.join(path, Model.META_FILE)
        self.name = self.path.split()[-1]

        self._meta: Optional[ModelMeta] = meta

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def id(self) -> str:
        return os.path.relpath(self.path, BASE_DIR)

    def meta(self) -> ModelMeta:
        if not self._meta:
            try:
                with open(self.meta_path, 'r') as f:
                    self._meta = ModelMeta.from_json(f.read())
            except FileNotFoundError:
                logger.warning("ModelMeta file not existing at {}. Creating a new one!".format(self.meta_path))
                self._meta = ModelMeta(
                    id=str(uuid.uuid4()),
                    created=datetime.datetime.now(),
                )

        self._meta.id = self.id()
        return self._meta

    def save_meta(self):
        if self._meta:
            self._meta.id = self.id()
            with open(self.meta_path, 'w') as f:
                f.write(self._meta.to_json(indent=2))

    def local_file(self, file: str) -> str:
        return os.path.join(self.path, file)

    def exists(self, file: str = None) -> bool:
        return os.path.exists(self.local_file(file if file else Model.META_FILE))

    def delete(self):
        if self.exists():
            shutil.rmtree(self.path)

    def copy_to(self, target_model: 'Model', override=True):
        if not self.exists():
            raise FileNotFoundError()

        if not override and target_model.exists():
            raise FileExistsError()

        copyied_model = Model(target_model.path, meta=self.meta())
        self._meta = None
        shutil.rmtree(target_model.path)
        shutil.copytree(self.path, target_model.path)
        copyied_model.save_meta()

