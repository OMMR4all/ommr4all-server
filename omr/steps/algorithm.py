import datetime
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from database import DatabaseBook, DatabasePage
from database.file_formats import PcGts
from database.file_formats.performance import LockState
from omr.dataset import DatasetCallback, Dataset
from typing import Optional, List, Type, Union, Generator
from omr.experimenter.experimenter import Experimenter

from .algorithmtrainerparams import AlgorithmTrainerSettings, AlgorithmTrainerParams, DatasetParams
from .algorithmpreditorparams import AlgorithmPredictorSettings, AlgorithmPredictorParams
from database.model import Models, Model, ModelMeta, ModelTrainingBook, ModelTrainingInfo, MetaId, ModelsId, Storage
from database.database_available_models import DatabaseAvailableModels
import os
import uuid
from .algorithmtypes import AlgorithmTypes, AlgorithmGroups, WorkerResource


class TrainerCallback(DatasetCallback, ABC):
    def __init__(self):
        super().__init__()
        self.total_iters = 0
        self.early_stopping_iters = 0

    def init(self, total_iters, early_stopping_iters):
        self.total_iters = total_iters
        self.early_stopping_iters = early_stopping_iters

    @abstractmethod
    def next_iteration(self, iter: int, loss: float, acc: float):
        pass

    @abstractmethod
    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        pass

    @abstractmethod
    def early_stopping(self):
        pass

    @abstractmethod
    def resolving_files(self):
        pass


class PredictionCallback(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def progress_updated(self,
                         percentage: float,
                         n_pages: int = 0,
                         n_processed_pages: int = 0):
        pass


class PredictionProgress:
    """Page-normalised, monotonic progress reporting for algorithm predictors.

    Every predictor reports through this class instead of calling
    ``PredictionCallback.progress_updated`` directly, so the invariants live in
    one place:

    * ``percentage`` is ``(pages_done + fraction_within_current_page) / n_pages``,
      clamped to ``[0, 1]`` and never decreasing,
    * ``n_processed_pages`` counts *finished pages only* -- it is never fed a
      sub-page counter, so the "Progress n/N" label in the client cannot jump
      back to 0 mid-run.

    Predictors that iterate lines rather than pages pass ``item_pages``, a list
    mapping item index -> page index. ``item_finished`` then advances the smooth
    line-granular percentage while still reporting true page counts.

    Constructing with ``callback=None`` yields a fully functional no-op, so
    predictors do not need ``if callback:`` guards around every call site.
    """

    def __init__(self,
                 callback: Optional[PredictionCallback],
                 n_pages: int,
                 item_pages: Optional[List[int]] = None):
        self.callback = callback
        self.n_pages = max(1, n_pages)
        self.item_pages = item_pages
        self._page_totals = Counter(item_pages) if item_pages is not None else Counter()
        self._page_remaining = Counter(self._page_totals)
        # Pages that contribute no items (e.g. a page without any text line) are
        # never reported by item_finished, so credit them up front -- otherwise
        # the label could never reach n/n.
        self._pages_done = (self.n_pages - len(self._page_totals)
                            if item_pages is not None else 0)
        self._fraction = 0.0
        self._last_percentage = 0.0

    @classmethod
    def for_lines(cls,
                  callback: Optional[PredictionCallback],
                  pcgts_files: list,
                  lines: list) -> 'PredictionProgress':
        """Reporter for predictors that iterate lines instead of pages.

        ``lines`` are the loaded ``RegionLineMaskData`` items; each carries the
        PcGts of the page it came from, which is what maps a line back to a page.
        """
        page_of = {id(f): i for i, f in enumerate(pcgts_files)}
        item_pages = [page_of.get(id(line.operation.pcgts), 0) for line in lines]
        return cls(callback, len(pcgts_files), item_pages)

    def start(self):
        """Publish an initial 0 / n_pages so the bar starts determinate."""
        self._emit()

    def sub_progress(self, fraction: float):
        """Report progress *within* the page currently being processed."""
        self._fraction = min(max(fraction, 0.0), 1.0)
        self._emit()

    def page_finished(self):
        """Advance to the next page. The only thing that moves n_processed_pages."""
        self._pages_done = min(self._pages_done + 1, self.n_pages)
        self._fraction = 0.0
        self._emit()

    def item_finished(self, index: int):
        """Report completion of item ``index`` for line-based predictors.

        Advances the percentage smoothly per line and rolls the page counter
        over once the last line of a page has been processed.
        """
        if self.item_pages is None or index >= len(self.item_pages):
            # No item -> page map available: fall back to treating items as pages.
            self.page_finished()
            return

        page = self.item_pages[index]
        self._page_remaining[page] -= 1
        if self._page_remaining[page] <= 0:
            self.page_finished()
            return

        total = self._page_totals[page]
        self._fraction = (total - self._page_remaining[page]) / total
        self._emit()

    def _emit(self):
        if not self.callback:
            return
        percentage = (self._pages_done + self._fraction) / self.n_pages
        percentage = min(max(percentage, 0.0), 1.0)
        # Monotonic: a slower sub-page estimate must never drag the bar back.
        percentage = max(percentage, self._last_percentage)
        self._last_percentage = percentage
        self.callback.progress_updated(percentage,
                                       n_pages=self.n_pages,
                                       n_processed_pages=self._pages_done)


class AlgorithmTrainer(ABC):
    @staticmethod
    @abstractmethod
    def meta() -> Type['AlgorithmMeta']:
        pass

    @staticmethod
    @abstractmethod
    def default_params() -> AlgorithmTrainerParams:
        pass

    @staticmethod
    def default_dataset_params() -> DatasetParams:
        return DatasetParams()

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        pass

    @staticmethod
    @abstractmethod
    def required_locks() -> List[LockState]:
        return []

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__()
        self.settings: AlgorithmTrainerSettings = settings
        if not self.settings.params:
            self.settings.params = self.__class__.default_params()
        else:
            self.settings.params.mix_default(self.__class__.default_params())

        self.settings.dataset_params.mix_default(self.__class__.default_dataset_params())
        self.__class__.force_dataset_params(self.settings.dataset_params)

        self.params: AlgorithmTrainerParams = self.settings.params

        self.train_dataset = self.meta().dataset_class()(self.settings.train_data, self.settings.dataset_params)
        self.validation_dataset = self.meta().dataset_class()(self.settings.validation_data,
                                                              self.settings.dataset_params)

    def train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        class CallbackInterception(TrainerCallback):
            def __init__(self, trainer: AlgorithmTrainer):
                super().__init__()
                self.model = trainer.settings.model
                self.meta = self.model.meta()
                self.init(trainer.params.n_iter, trainer.params.early_stopping_max_keep)

            def init(self, total_iters, early_stopping_iters):
                if callback:
                    callback.init(total_iters, early_stopping_iters)

            def next_iteration(self, iter: int, loss: float, acc: float):
                if callback:
                    callback.next_iteration(iter, loss, acc)

            def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
                if callback:
                    callback.next_best_model(best_iter, best_acc, best_iters)

                self.meta.accuracy = best_acc
                self.meta.iters = best_iter
                self.model.save_meta()

            def early_stopping(self):
                if callback:
                    callback.early_stopping()

            def resolving_files(self):
                if callback:
                    callback.resolving_files()

            def loading(self, n: int, total: int):
                if callback:
                    callback.loading(n, total)

            def loading_started(self, total: int):
                if callback:
                    callback.loading_started(total)

            def loading_finished(self, total: int):
                if callback:
                    callback.loading_finished(total)

        if not self.settings.model:
            if target_book:
                self.settings.model = self.meta().create_new_model(target_book)
            else:
                raise ValueError()

        self.settings.model.save_meta()

        # written before the run so that an aborted training still says what it was fed
        training_info = self._training_info(target_book)
        self.settings.model.save_training_info(training_info)

        self._pre_train()
        self._train(target_book, CallbackInterception(self))
        self._post_train(target_book)

        self.settings.model.save_meta()
        training_info.finished = datetime.datetime.now()
        self.settings.model.save_training_info(training_info)

    def _training_info(self, target_book: Optional[DatabaseBook] = None) -> ModelTrainingInfo:
        """The configuration and the ground truth of this run, for the model's training.json."""
        books: 'OrderedDict[str, ModelTrainingBook]' = OrderedDict()

        def add(data: List[PcGts], validation: bool):
            for pcgts in data:
                page = pcgts.page.location
                entry = books.get(page.book.book)
                if entry is None:
                    entry = ModelTrainingBook(book=page.book.book,
                                              book_name=page.book.get_meta().name)
                    books[page.book.book] = entry
                (entry.validation_pages if validation else entry.train_pages).append(page.page)

        add(self.settings.train_data, False)
        add(self.settings.validation_data, True)
        for entry in books.values():
            entry.train_pages.sort()
            entry.validation_pages.sort()

        return ModelTrainingInfo(
            algorithm_type=self.meta().type().value,
            target_book=target_book.book if target_book else None,
            started=datetime.datetime.now(),
            started_by=self.settings.started_by,
            pretrained_model=self.params.load if self.params else None,
            n_train=self.settings.n_train,
            n_epoch=self.params.n_epoch if self.params else None,
            params=self.params.to_dict() if self.params else None,
            dataset_params=self.settings.dataset_params.to_dict() if self.settings.dataset_params else None,
            books=list(books.values()),
            n_train_pages=len(self.settings.train_data),
            n_validation_pages=len(self.settings.validation_data),
        )

    @abstractmethod
    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        return None

    def _pre_train(self):
        with open(self.settings.model.local_file("dataset_params.json"), 'w') as f:
            f.write(self.settings.dataset_params.to_json())

    def _post_train(self, target_book: Optional[DatabaseBook] = None):
        with open(self.settings.model.local_file("dataset_params.json"), 'w') as f:
            f.write(self.settings.dataset_params.to_json())


class AlgorithmPredictionResult(ABC):
    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def store_to_page(self):
        pass


AlgorithmPredictionResultGenerator = Generator[AlgorithmPredictionResult, None, None]


class AlgorithmPredictor(ABC):
    @staticmethod
    @abstractmethod
    def meta() -> Type['AlgorithmMeta']:
        pass

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__()
        self.settings: AlgorithmPredictorSettings = settings
        self.params: AlgorithmPredictorParams = settings.params

        if self.params.modelId:
            # override model if an id is given
            self.settings.model = Model(self.params.modelId)

        try:
            if not settings.model:
                raise ValueError("Model may not be None")
            with open(settings.model.local_file('dataset_params.json'), 'r') as f:
                self.dataset_params = DatasetParams.from_json(f.read())
        except FileNotFoundError:
            self.dataset_params = DatasetParams()

    @abstractmethod
    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        pass

    @classmethod
    @abstractmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        pass
    @classmethod
    def unlocked(cls, page: DatabasePage) -> bool:
        lock = cls.meta().group().group_2_lock_mapping()
        if lock is None:
            # Groups without a lock (preprocessing, tools, postprocessing) do not overwrite
            # anything a user can lock, so every page stays available to them.
            return True
        return not page.page_progress().locked.get(lock)


class AlgorithmMeta(ABC):

    @staticmethod
    @abstractmethod
    def type() -> AlgorithmTypes:
        pass

    @classmethod
    def group(cls) -> AlgorithmGroups:
        return cls.type().group()

    @classmethod
    @abstractmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        pass

    @classmethod
    @abstractmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        pass

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        pass

    # Worker resource policy: which resource class (CPU/GPU worker) tasks of
    # this algorithm run on by default and are allowed to run on. The request
    # may override the default, but only within the allowed set.
    @classmethod
    def default_predictor_resource(cls) -> WorkerResource:
        return WorkerResource.CPU

    @classmethod
    def allowed_predictor_resources(cls) -> List[WorkerResource]:
        return [WorkerResource.CPU, WorkerResource.GPU]

    @classmethod
    def default_trainer_resource(cls) -> WorkerResource:
        return WorkerResource.GPU

    @classmethod
    def allowed_trainer_resources(cls) -> List[WorkerResource]:
        return [WorkerResource.CPU, WorkerResource.GPU]

    @classmethod
    def create_trainer(cls, settings: AlgorithmTrainerSettings) -> AlgorithmTrainer:
        return cls.trainer()(settings)

    @classmethod
    def create_predictor(cls, settings) -> AlgorithmPredictor:
        return cls.predictor()(settings)

    @staticmethod
    @abstractmethod
    def dataset_class() -> Type[Dataset]:
        pass

    @classmethod
    def model_dir(cls) -> str:
        return cls.type().model_type().value

    @classmethod
    def models_for_book(cls, book: DatabaseBook) -> Models:
        return Models(ModelsId.from_external(book.book, cls.type()))

    @classmethod
    def default_model_for_book(cls, book: DatabaseBook) -> Model:
        models = ModelsId.from_internal(book.get_meta().notationStyle, cls.type())
        return Model(MetaId(models, cls.model_dir()))

    @classmethod
    def default_model_for_style(cls, style: str) -> Optional[Model]:
        models = ModelsId.from_internal(style, cls.type())
        model = Model(MetaId(models, cls.model_dir()))
        if model.exists():
            return model

        # fallback: french14 must exist
        return cls.default_model_for_style('french14')

    @classmethod
    def newest_model_for_book(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        if not book:
            return None

        best_model = cls.models_for_book(book).newest_model()
        if best_model and best_model.exists():
            return best_model

        return None

    @classmethod
    def best_model_for_book(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        newest_model = cls.newest_model_for_book(book)
        if newest_model and newest_model.exists():
            return newest_model
        return cls.default_model_for_book(book)

    @classmethod
    def selected_algorithm_params_for_book(cls, book: Optional[DatabaseBook]) -> Optional[AlgorithmPredictorParams]:
        return None if not book else book.get_meta().algorithm_predictor_params(cls.type())

    @classmethod
    def selected_model_for_book(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        selected_params = cls.selected_algorithm_params_for_book(book)
        if selected_params and selected_params.modelId:
            model = Model(selected_params.modelId)
            if model and model.exists():
                return model
        best = cls.best_model_for_book(book)
        if best and best.exists():
            return best
        # fallback: french14 must exist
        return cls.default_model_for_style('french14')

    @classmethod
    def model_of_book_style(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        if not book:
            return None

        model = cls.default_model_for_style(book.get_meta().notationStyle)
        if model and model.exists():
            return model

        return None

    @classmethod
    def create_new_model(cls, book: DatabaseBook, id: Optional[str] = None) -> Model:
        import datetime

        id = id if id else str(uuid.uuid4())
        time = datetime.datetime.now()
        models = ModelsId.from_external(book.book, cls.type())
        return Model(MetaId(models, time.strftime("%Y-%m-%dT%H:%M:%S")),
                     ModelMeta(id,
                               time,
                               style=book.get_meta().notationStyle)
                     )

    @classmethod
    def list_available_models_for_style(cls, style: str) -> DatabaseAvailableModels:
        default_style_model = cls.default_model_for_style(style).meta() if cls.default_model_for_style(style) else None
        return DatabaseAvailableModels(
            selected_model=default_style_model,
            default_book_style_model=default_style_model,
            models_of_same_book_style=[(b.get_meta(), cls.newest_model_for_book(b).meta()) for b in
                                       DatabaseBook.list_available_of_style(style) if cls.newest_model_for_book(b)]
        )
