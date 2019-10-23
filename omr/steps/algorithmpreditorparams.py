from dataclasses import dataclass, field
from database.model import Model, MetaId
from mashumaro import DataClassJSONMixin
from mashumaro.types import SerializableType
from typing import Optional, TYPE_CHECKING
from .algorithmtypes import AlgorithmTypes
from database.file_formats.pcgts import Coords
from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoderParams
from google.protobuf.json_format import MessageToDict, ParseDict


class SerializableCTCDecoderParams(SerializableType):
    def __init__(self,
                 type=CTCDecoderParams.CTC_WORD_BEAM_SEARCH,
                 beam_width=50):
        self.params = CTCDecoderParams()
        self.params.type = type
        self.params.beam_width = beam_width

    def _serialize(self):
        return MessageToDict(self.params)

    @classmethod
    def _deserialize(cls, value):
        params = SerializableCTCDecoderParams()
        params.params = ParseDict(value, CTCDecoderParams())
        return params


@dataclass()
class AlgorithmPredictorParams(DataClassJSONMixin):
    # general
    modelId: Optional[MetaId] = None   # This field can override the model specified in the AlgorithmPredictorSettings

    # preprocessing
    automaticLd: bool = True
    avgLd: int = 10

    # staff line detection
    minNumberOfStaffLines: Optional[int] = None
    maxNumberOfStaffLines: Optional[int] = None

    # ocr
    ctcDecoder: SerializableCTCDecoderParams = field(default_factory=lambda: SerializableCTCDecoderParams())

    # tools
    # layout connected components
    initialLine: Optional['Coords'] = None


@dataclass()
class AlgorithmPredictorSettings:
    model: Model
    params: AlgorithmPredictorParams = field(default_factory=lambda: AlgorithmPredictorParams())

