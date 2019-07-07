from dataclasses import dataclass, field
from database.model import Model
from mashumaro import DataClassJSONMixin
from typing import Optional
from enum import Enum
from .algorithmtypes import AlgorithmTypes


class LayoutModes(Enum):
    SIMPLE = 'simple'
    COMPLEX = 'complex'

    def to_predictor_type(self) -> AlgorithmTypes:
        return {
            LayoutModes.SIMPLE: AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES,
            LayoutModes.COMPLEX: AlgorithmTypes.LAYOUT_COMPLEX_STANDARD,
        }[self]


@dataclass()
class AlgorithmPredictorParams(DataClassJSONMixin):
    # general
    modelId: Optional[str] = None   # This field can override the model specified in the AlgorithmPredictorSettings

    # preprocessing
    automaticLd: bool = True
    avgLd: int = 10

    # layout
    layoutMode: LayoutModes = LayoutModes.COMPLEX


@dataclass()
class AlgorithmPredictorSettings:
    model: Model
    params: AlgorithmPredictorParams = field(default_factory=lambda: AlgorithmPredictorParams())

