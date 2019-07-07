from dataclasses import dataclass, field
from mashumaro import DataClassJSONMixin
import datetime


@dataclass()
class ModelMeta(DataClassJSONMixin):
    id: str = None
    created: datetime.datetime = field(default_factory=lambda: datetime.datetime.now())
    accuracy: float = 0
    iters: int = 0
