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
    # None keeps the algorithm default; the value is clamped per user in workerresources.resolve_n_epoch
    n_epoch: Optional[int] = None

    def to_trainer_params(self, trainer_class) -> Optional['AlgorithmTrainerParams']:
        """The hyper parameters requested by the client, or None to keep the algorithm defaults.

        Built from the algorithm's own default_params() rather than from a bare
        AlgorithmTrainerParams: mix_default() only fills in fields that are None or negative, so a
        fresh instance would silently override defaults such as ``display`` that are positive.
        """
        if self.n_epoch is None:
            return None
        params = trainer_class.default_params()
        params.n_epoch = self.n_epoch
        return params

    def to_train_val(self, locks: List[LockState], shuffle: bool = True, books: List[DatabaseBook] = None) -> Tuple[List[PcGts], List[PcGts]]:
        if self.includeAllTrainingData:
            books = DatabaseBook.list_available()

        return dataset_by_locked_pages(self.nTrain, locks, shuffle, books)

