from dataclasses import dataclass, field
from dataclasses import dataclass, field as datafields
from database.file_formats.book.document import Document
from database.model import Model, MetaId
from mashumaro import DataClassJSONMixin
from mashumaro.types import SerializableType
from typing import Optional, TYPE_CHECKING, List
from .algorithmtypes import AlgorithmTypes
from database.file_formats.pcgts import Coords
from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import CTCDecoderParams, CTCDecoderType
from google.protobuf.json_format import MessageToDict, ParseDict

def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = {f.name:f.type for f in datafields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except:
        return dikt

@dataclass
class SerializableCTCDecoderParams(CTCDecoderParams, DataClassJSONMixin):
    type: CTCDecoderType = CTCDecoderType.Default
    blank_index: int = 0
    min_p_threshold: float = 0

    beam_width = 25
    non_word_chars: List[str] = field(default_factory=lambda: list("0123456789[]()_.:;!?{}-'\""))

    dictionary: List[str] = field(default_factory=list)
    word_separator: str = " "
   # def _serialize(self):
   #     dataclass_from_dict(SerializableCTCDecoderParams, self.params)
   #     return MessageToDict(self.params)

   # @classmethod
   # def _deserialize(cls, value):
   #     params = SerializableCTCDecoderParams()
   #     params.params = ParseDict(value, CTCDecoderParams())
   #     return params


@dataclass()
class AlgorithmPredictorParams(DataClassJSONMixin):
    # general
    modelId: Optional[MetaId] = None   # This field can override the model specified in the AlgorithmPredictorSettings

    # preprocessing
    automaticLd: bool = True
    avgLd: int = 10
    deskew: bool = True

    # layout
    dropCapitals: bool = True
    documentStarts: bool = True
    documentStartsDropCapitalMinHeight: float = 0.5

    # staff line detection
    minNumberOfStaffLines: Optional[int] = None
    maxNumberOfStaffLines: Optional[int] = None

    # ocr
    ctcDecoder: SerializableCTCDecoderParams = field(default_factory=lambda: SerializableCTCDecoderParams())

    # tools
    # symbol predicition
    use_clef_pos_correction = True

    # layout connected components
    initialLine: Optional['Coords'] = None

    documentId: str = None
    documentText: str = None

    useDictionaryCorrection: bool = True


@dataclass()
class AlgorithmPredictorSettings:
    model: Model
    params: AlgorithmPredictorParams = field(default_factory=lambda: AlgorithmPredictorParams())

