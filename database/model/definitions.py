from enum import Enum
import os
from ommr4all.settings import BASE_DIR, PRIVATE_MEDIA_ROOT
from typing import Optional, NamedTuple, List, Mapping
from omr.steps.algorithmtypes import AlgorithmTypes
from mashumaro.types import SerializableType
from mashumaro import DataClassDictMixin
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class StorageType(Enum):
    INTERNAL = 'i'
    EXTERNAL = 'e'
    CUSTOM = 'c'

    def path(self):
        return {
            StorageType.INTERNAL: os.path.join(BASE_DIR, 'internal_storage'),
            StorageType.EXTERNAL: PRIVATE_MEDIA_ROOT,
            StorageType.CUSTOM: '',
        }[self]


class Storage(NamedTuple):
    type: StorageType
    custom_path: Optional[str] = None
    
    def path(self) -> str:
        if self.type == StorageType.CUSTOM:
            return self.custom_path
        
        return self.type.path()
    
    @staticmethod
    def custom(custom_path: str):
        return Storage(StorageType.CUSTOM, custom_path)


class ModelsId(NamedTuple):
    @staticmethod
    def from_internal(notation_style: str, algorithm_type: AlgorithmTypes):
        return ModelsId(Storage(StorageType.INTERNAL), None, notation_style, algorithm_type.model_type())

    @staticmethod
    def from_external(book: str, algorithm_type: AlgorithmTypes):
        return ModelsId(Storage(StorageType.EXTERNAL), book, None, algorithm_type.model_type())

    @staticmethod
    def parse(s: str, remaining: Optional[List[str]]) -> 'ModelsId':
        s = s.split('/')
        storage_type = StorageType(s[0])
        if storage_type == StorageType.CUSTOM:
            remaining.append(s[-1])
            return ModelsId(
                Storage(storage_type,
                        '/'.join(s[2:-1]),
                        ),
                None,
                None,
                AlgorithmTypes(s[1]),
            )
        else:
            if remaining is not None:
                remaining += s[3:]
            return ModelsId(
                Storage(storage_type),
                s[1] if storage_type == StorageType.EXTERNAL else None,
                s[1] if storage_type == StorageType.INTERNAL else None,
                AlgorithmTypes(s[2]),
            )

    storage: Storage
    book: Optional[str]
    notation_style: Optional[str]
    algorithm_type: AlgorithmTypes

    def path(self):
        from omr.steps.step import Step
        if self.storage.type == StorageType.EXTERNAL:
            return os.path.join(self.storage.path(), self.book, 'models', Step.create_meta(self.algorithm_type).model_dir())
        elif self.storage.type == StorageType.INTERNAL:
            return os.path.join(self.storage.path(), 'default_models', self.notation_style)
        else:
            return self.storage.path()

    def __str__(self):
        if self.storage.type == StorageType.EXTERNAL:
            return '/'.join([self.storage.type.value, self.book, self.algorithm_type.value])
        elif self.storage.type == StorageType.INTERNAL:
            return '/'.join([self.storage.type.value, self.notation_style, self.algorithm_type.value])
        else:
            return '/'.join([self.storage.type.value, self.algorithm_type.value, self.storage.custom_path])


@dataclass
class MetaId(SerializableType):
    models: ModelsId
    name: str

    def path(self):
        return os.path.join(self.models.path(), self.name)

    def __str__(self):
        return '/'.join([str(self.models), self.name])

    @staticmethod
    def from_str(s: str) -> 'MetaId':
        remaining = []
        models = ModelsId.parse(s, remaining)
        return MetaId(
            models,
            remaining[0],
        )
    
    @staticmethod
    def from_custom_path(s: str, algorithm_type: AlgorithmTypes) -> 'MetaId':
        base, name = os.path.split(s)
        return MetaId(ModelsId(Storage.custom(base), None, None, algorithm_type), name)

    def _serialize(self) -> str:
        return str(self)

    @classmethod
    def _deserialize(cls, value: str) -> Optional['MetaId']:
        try:
            return MetaId.from_str(value)
        except Exception as e:
            logger.error('Could not parse MetaId from {}'.format(value))
            logger.exception(e)
            return None

    def to_dict(
            self,
            use_bytes: bool = False,
            use_enum: bool = False,
            use_datetime: bool = False):
        return self._serialize()

    @classmethod
    def from_dict(
            cls,
            d: Mapping,
            use_bytes: bool = False,
            use_enum: bool = False,
            use_datetime: bool = False):
        return cls._deserialize(d)



