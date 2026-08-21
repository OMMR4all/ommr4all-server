from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Dataset, \
    Model, Optional, DatabaseBook
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.STAFF_LINES_CORRECTION

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import StaffLineCorrectionPredictor
        return StaffLineCorrectionPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return None

    @classmethod
    def selected_model_for_book(cls, book: Optional[DatabaseBook]) -> Optional[Model]:
        # This step is purely image based and loads no model. The inherited lookup would
        # walk the default model directories and, for a step that never ships a model,
        # recurse into default_model_for_style('french14') forever. Returning the (possibly
        # non-existent) placeholder keeps AlgorithmPredictor.__init__ happy: it only opens
        # dataset_params.json and falls back to the defaults if that file is missing.
        return cls.default_model_for_book(book) if book else None


Step.register(Meta)
