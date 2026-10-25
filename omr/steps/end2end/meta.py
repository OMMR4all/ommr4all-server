from abc import abstractmethod
from typing import Optional, Type

from database import DatabaseBook
from database.model import Model, MetaId, ModelsId
from omr.steps.algorithm import AlgorithmMeta, Experimenter

from .dataset import End2EndDataset, Dataset
from .networks import End2EndNetwork


class End2EndMeta(AlgorithmMeta):
    """Shared by all end-to-end architectures; a subclass names its network."""

    @classmethod
    @abstractmethod
    def create_network(cls, vocab_size: int, localization: bool, pretrained_backbone: bool,
                       loc_feedback: bool = False) -> End2EndNetwork:
        pass

    @classmethod
    @abstractmethod
    def default_height(cls) -> int:
        """Input height of the network in pixels."""
        pass

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return End2EndDataset

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from .experimenter import End2EndExperimenter
        return End2EndExperimenter

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
