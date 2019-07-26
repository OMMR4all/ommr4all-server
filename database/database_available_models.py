from dataclasses import dataclass, field
from mashumaro import DataClassDictMixin
from typing import List, Optional, Tuple, NamedTuple
from database.model import ModelMeta
from database.database_book_meta import DatabaseBookMeta
from ommr4all.settings import BASE_DIR
import os

@dataclass
class DefaultModel(DataClassDictMixin):
    style: str
    model: ModelMeta

@dataclass
class DatabaseAvailableModels(DataClassDictMixin):
    book: str
    book_meta: Optional[DatabaseBookMeta] = None
    newest_model: Optional[ModelMeta] = None
    selected_model: Optional[ModelMeta] = None
    book_models: List[ModelMeta] = field(default_factory=lambda: [])
    default_book_style_model: Optional[ModelMeta] = None
    default_models: List[DefaultModel] = field(default_factory=lambda: [])
    models_of_same_book_style: List[Tuple[DatabaseBookMeta, ModelMeta]] = field(default_factory=lambda: [])

    @staticmethod
    def local_default_model_path_for_style(style: str, sub=''):
        return os.path.join(BASE_DIR, 'internal_storage', 'default_models', style, sub)

