from dataclasses import dataclass, field
#from mashumaro import DataClassJSONMixin
import datetime

from mashumaro import field_options
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass()
class ModelMeta(DataClassJSONMixin):
    id: str = None
    created: datetime.datetime = field(default_factory=lambda: datetime.datetime.now())
    accuracy: float = field(default=0.0, metadata=field_options(serialize=float)
    )
    iters: int = 0
    style: str = 'french14'
