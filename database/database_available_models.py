from dataclasses import dataclass, field
from mashumaro import DataClassJSONMixin
from typing import List, Optional, Tuple
from database.model import ModelMeta
from database.database_book_meta import DatabaseBookMeta


@dataclass()
class DatabaseAvailableModels(DataClassJSONMixin):
    book: str
    book_meta: Optional[DatabaseBookMeta] = None
    newest_model: Optional[ModelMeta] = None
    selected_model: Optional[ModelMeta] = None
    book_models: List[ModelMeta] = field(default_factory=lambda: [])
    default_book_style_model: Optional[ModelMeta] = None
    models_of_same_book_style: List[Tuple[DatabaseBookMeta, ModelMeta]] = field(default_factory=lambda: [])
