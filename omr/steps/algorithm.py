from abc import ABC, abstractmethod
from database import DatabaseBook, DatabasePage
from database.file_formats import PcGts
from database.file_formats.performance import LockState
from omr.dataset import DatasetCallback, Dataset
from typing import Optional, List, Type, Union, Generator
from omr.experimenter.experimenter import Experimenter
from .algorithmtrainerparams import AlgorithmTrainerSettings, AlgorithmTrainerParams, DatasetParams
from .algorithmpreditorparams import AlgorithmPredictorSettings, AlgorithmPredictorParams
from database.model import Models, Model, ModelMeta, MetaId, ModelsId, Storage
from database.database_available_models import DatabaseAvailableModels
import os
import uuid
from .algorithmtypes import AlgorithmTypes, AlgorithmGroups


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

        self._pre_train()
        self._train(target_book, CallbackInterception(self))
        self._post_train(target_book)

        self.settings.model.save_meta()

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
        return not page.page_progress().locked.get(cls.meta().group().group_2_lock_mapping())


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
