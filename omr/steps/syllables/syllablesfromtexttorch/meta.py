from typing import Optional

from database.model import Model, MetaId, ModelsId
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Dataset, Experimenter
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH

    @classmethod
    def model_dir(cls) -> str:
        """Own directory for the *default* model of this step.

        The step runs an OCR model and therefore shares the trained models of a book with
        text_guppy (AlgorithmTypes.model_type(), applied by ModelsId.from_external, which then
        asks the Guppy meta for the directory). Only the default model per notation style lives
        here, so that a style can use a different model for syllables than for text.
        """
        return AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH.value

    @classmethod
    def default_model_for_book(cls, book) -> Model:
        # via default_model_for_style, so that the fallback below applies here as well: the base
        # implementation would hand out the bare directory that ships with the repository
        return cls.default_model_for_style(book.get_meta().notationStyle)

    @classmethod
    def default_model_for_style(cls, style: str) -> Optional[Model]:
        # No default is shipped for this step: the base implementation would recurse forever on
        # its french14 fallback. Without an own default the text model of the style is used,
        # which is what this step ran on before it had a slot of its own.
        model = Model(MetaId(ModelsId.from_internal(style, cls.type()), cls.model_dir()))
        # has_weights, not exists: the directory is shipped with a bare meta.json
        if model.exists() and model.has_weights():
            return model
        return Step.meta(AlgorithmTypes.OCR_GUPPY).default_model_for_style(style)

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import SyllablesFromTextPredictor
        return SyllablesFromTextPredictor

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import SyllablesExperimenter
        return SyllablesExperimenter

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return None


Step.register(Meta)
