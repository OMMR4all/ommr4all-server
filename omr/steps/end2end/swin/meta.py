from omr.steps.algorithm import AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step

from ..meta import End2EndMeta
from ..networks import End2EndNetwork, SwinTransformerOMR


class Meta(End2EndMeta):
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

    @classmethod
    def create_network(cls, vocab_size: int, localization: bool, pretrained_backbone: bool,
                       loc_feedback: bool = False) -> End2EndNetwork:
        # the Swin backbone is always initialised from its pretrained weights
        return SwinTransformerOMR(vocab_size=vocab_size, localization=localization, loc_feedback=loc_feedback)

    @classmethod
    def default_height(cls) -> int:
        return 224


Step.register(Meta)
