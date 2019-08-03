from dataclasses import dataclass, field
from database.model import Model
from mashumaro import DataClassJSONMixin
from typing import Optional, TYPE_CHECKING
from .algorithmtypes import AlgorithmTypes
from database.file_formats.pcgts import Coords


@dataclass()
class AlgorithmPredictorParams(DataClassJSONMixin):
    # general
    modelId: Optional[str] = None   # This field can override the model specified in the AlgorithmPredictorSettings

    # preprocessing
    automaticLd: bool = True
    avgLd: int = 10

    # tools
    # layout connected components
    initialLine: Optional['Coords'] = None


@dataclass()
class AlgorithmPredictorSettings:
    model: Model
    params: AlgorithmPredictorParams = field(default_factory=lambda: AlgorithmPredictorParams())

