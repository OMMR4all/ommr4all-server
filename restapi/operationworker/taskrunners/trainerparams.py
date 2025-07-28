from mashumaro.mixins.json import DataClassJSONMixin
from dataclasses import dataclass
from typing import Tuple, List, Optional
from database import DatabaseBook
from database.file_formats import PcGts
from omr.dataset.datafiles import dataset_by_locked_pages, LockState
from database.model import ModelMeta


@dataclass()
class TaskTrainerParams(DataClassJSONMixin):
    nTrain: float = 0.8
    includeAllTrainingData: bool = False
    pretrainedModel: Optional[ModelMeta] = None
    symbol_enable_neume_training: bool = False
    symbol_enable_additional_symbol_types: bool = False

    def to_train_val(self, locks: List[LockState], shuffle: bool = True, books: List[DatabaseBook] = None) -> Tuple[List[PcGts], List[PcGts]]:
        if self.includeAllTrainingData:
            books = DatabaseBook.list_available()

        return dataset_by_locked_pages(self.nTrain, locks, shuffle, books)

