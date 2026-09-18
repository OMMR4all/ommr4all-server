"""Interface of neume grouping methods."""
from abc import ABC, abstractmethod
from typing import List

from omr.discovery.config import GroupingConfig
from omr.discovery.schema import SymbolCandidate, SymbolRelation


class NeumeGroupingMethod(ABC):
    name = 'base'

    @abstractmethod
    def group(self, crop, symbols: List[SymbolCandidate], cfg: GroupingConfig) -> List[List[str]]: ...

    @abstractmethod
    def relations(self, crop, symbols: List[SymbolCandidate], group: List[str],
                  cfg: GroupingConfig) -> List[SymbolRelation]: ...
