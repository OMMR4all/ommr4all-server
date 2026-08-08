from dataclasses import dataclass, field
#from mashumaro import DataClassJSONMixin
import datetime
from typing import Optional

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
    # id of the model this one was copied from (Model.copy_to), e.g. the book model a style
    # default was created from. None for trained models and for metas written before this field.
    source_id: Optional[str] = None


@dataclass()
class ModelUsage(DataClassJSONMixin):
    """When a model was last loaded for a prediction, stored next to it in usage.json.

    Kept out of meta.json on purpose: the predictor cache keys on the modification time of
    meta.json, so writing the usage there would drop the cached predictor on every prediction.
    """
    last_used: Optional[datetime.datetime] = None
    n_used: int = 0
