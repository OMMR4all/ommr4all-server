from enum import Enum
import os
from ommr4all.settings import BASE_DIR, PRIVATE_MEDIA_ROOT
from typing import Optional, NamedTuple, List
from omr.steps.algorithmtypes import AlgorithmTypes


class Storage(Enum):
    INTERNAL = 'i'
    EXTERNAL = 'e'

    def path(self):
        return {
            Storage.INTERNAL: os.path.join(BASE_DIR, 'internal_storage'),
            Storage.EXTERNAL: PRIVATE_MEDIA_ROOT,
        }[self]


class ModelsId(NamedTuple):
    @staticmethod
    def from_internal(notation_style: str, algorithm_type: AlgorithmTypes):
        return ModelsId(Storage.INTERNAL, None, notation_style, algorithm_type)

    @staticmethod
    def from_external(book: str, algorithm_type: AlgorithmTypes):
        return ModelsId(Storage.EXTERNAL, book, None, algorithm_type)

    @staticmethod
    def parse(s: str, remaining: Optional[List[str]]) -> 'ModelsId':
        s = s.split('/')
        if remaining is not None:
            remaining += s[3:]
        storage = Storage(s[0])
        return ModelsId(
            storage,
            s[1] if storage == Storage.EXTERNAL else None,
            s[1] if storage == Storage.INTERNAL else None,
            AlgorithmTypes(s[2]),
        )

    storage: Storage
    book: Optional[str]
    notation_style: Optional[str]
    algorithm_type: AlgorithmTypes

    def path(self):
        from omr.steps.step import Step
        if self.storage == Storage.EXTERNAL:
            return os.path.join(self.storage.path(), self.book, 'models', Step.create_meta(self.algorithm_type).model_dir())
        else:
            return os.path.join(self.storage.path(), 'default_models', self.notation_style)

    def __str__(self):
        return '/'.join([self.storage.value, self.book if self.storage == Storage.EXTERNAL else self.notation_style, self.algorithm_type.value])


class MetaId(NamedTuple):
    models: ModelsId
    name: str

    def path(self):
        return os.path.join(self.models.path(), self.name)

    def __str__(self):
        return '/'.join([str(self.models), self.name])

    @staticmethod
    def from_str(s: str):
        remaining = []
        models = ModelsId.parse(s, remaining)
        return MetaId(
            models,
            remaining[0],
        )


