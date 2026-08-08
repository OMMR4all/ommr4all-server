from dataclasses import dataclass, field
#from mashumaro import DataClassJSONMixin
import datetime
from typing import Any, Dict, List, Optional

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


@dataclass()
class ModelTrainingBook(DataClassJSONMixin):
    """The ground truth one book contributed to a training run."""
    book: str
    book_name: Optional[str] = None
    train_pages: List[str] = field(default_factory=list)
    validation_pages: List[str] = field(default_factory=list)


@dataclass()
class ModelTrainingInfo(DataClassJSONMixin):
    """How a model was trained, stored next to it in training.json.

    Kept out of meta.json for the same reason as ModelUsage, and because the page lists would
    then be read by everything that only wants the accuracy of a model. The dataset parameters
    (augmentation and the like) are also written to dataset_params.json by the trainer; they are
    repeated here so that this file alone describes the run.
    """
    algorithm_type: Optional[str] = None
    target_book: Optional[str] = None
    started: Optional[datetime.datetime] = None
    finished: Optional[datetime.datetime] = None
    started_by: Optional[str] = None
    # id of the model the training started from, if any
    pretrained_model: Optional[str] = None
    # share of the ground truth used for training, the rest is validation
    n_train: Optional[float] = None
    n_epoch: Optional[int] = None
    params: Optional[Dict[str, Any]] = None
    dataset_params: Optional[Dict[str, Any]] = None
    books: List[ModelTrainingBook] = field(default_factory=list)
    n_train_pages: int = 0
    n_validation_pages: int = 0
