from omr.steps.algorithm import AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step

from ..meta import End2EndMeta
from ..networks import End2EndNetwork, ConvNeXtTransformerOMR

# input height in pixels of newly trained models; a trained model keeps its own (dataset_params.json).
# 320 instead of 256: on the full data -0.6 hSAR (noise) for +5 points of symbols within the detection
# distance and F1 0.808 -> 0.858 (docs/end2end-experiments.md, finding 14); ~1.7x training time
INPUT_HEIGHT = 320


class Meta(End2EndMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.END2END_CONVNEXT

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import ConvNeXtPredictor
        return ConvNeXtPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import ConvNeXtTrainer
        return ConvNeXtTrainer

    @classmethod
    def create_network(cls, vocab_size: int, localization: bool, pretrained_backbone: bool,
                       loc_feedback: bool = False) -> End2EndNetwork:
        return ConvNeXtTransformerOMR(vocab_size=vocab_size, localization=localization,
                                      pretrained_backbone=pretrained_backbone, loc_feedback=loc_feedback)

    @classmethod
    def default_height(cls) -> int:
        return INPUT_HEIGHT


Step.register(Meta)
