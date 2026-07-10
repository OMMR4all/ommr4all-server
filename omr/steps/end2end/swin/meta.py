from typing import Optional

from database import DatabaseBook
from database.model import Model, MetaId, ModelsId
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from omr.steps.step import Step

from ..dataset import End2EndDataset, Dataset


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.END2END_SWIN

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import SwinPredictor
        return SwinPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import SwinTrainer
        return SwinTrainer

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return End2EndDataset

    @classmethod
    def default_model_for_style(cls, style: str) -> Optional[Model]:
        # No internal default models exist for this algorithm; the base implementation
        # would recurse forever on its french14 fallback.
        models = ModelsId.from_internal(style, cls.type())
        model = Model(MetaId(models, cls.model_dir()))
        return model if model.exists() else None

    @classmethod
    def best_model_for_book(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        return cls.newest_model_for_book(book)

    @classmethod
    def selected_model_for_book(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        selected_params = cls.selected_algorithm_params_for_book(book)
        if selected_params and selected_params.modelId:
            model = Model(selected_params.modelId)
            if model and model.exists():
                return model
        return cls.best_model_for_book(book)


Step.register(Meta)
