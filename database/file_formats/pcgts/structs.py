from mashumaro import DataClassJSONMixin
from dataclasses import dataclass
from typing import List


@dataclass
class AnnotationSyllableConnectorStruct(DataClassJSONMixin):
    syllableID: str
    noteID: str


@dataclass
class AnnotationConnectionStruct(DataClassJSONMixin):
    musicID: str
    textID: str
    syllableConnectors: List[AnnotationSyllableConnectorStruct]


@dataclass
class AnnotationStruct(DataClassJSONMixin):
    connections: List[AnnotationConnectionStruct]
