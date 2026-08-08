from mashumaro.mixins.json import DataClassJSONMixin
from dataclasses import dataclass, field
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
    # names of the additional books to train on; only honoured together with includeAllTrainingData
    # and only after workerresources.validate_training_books checked the user may read them
    books: List[str] = field(default_factory=list)
    # who started the run, for the model's training.json. Set server side (see
    # restapi/views/bookoperations.py); anything the request body carries here is overwritten.
    started_by: Optional[str] = None

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
        """Ground truth of the trained book (``books``) plus the additionally selected books.

        The trained book always contributes: the client offers it pre-selected and locked, so
        ``self.books`` only carries the *extra* books. A request without an explicit selection
        keeps the historic meaning of the flag (every book on disk) so that older clients, which
        only send the boolean, behave as before.
        """
        if self.includeAllTrainingData:
            if self.books:
                books = list(books or [])
                selected = {b.book for b in books}
                for name in self.books:
                    if name not in selected:
                        selected.add(name)
                        books.append(DatabaseBook(name))
            else:
                books = DatabaseBook.list_available()

        return dataset_by_locked_pages(self.nTrain, locks, shuffle, books)

