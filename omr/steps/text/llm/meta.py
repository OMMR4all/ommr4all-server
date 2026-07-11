from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from omr.steps.algorithmtypes import WorkerResource
from ..dataset import TextDataset, Dataset
from omr.steps.step import Step
from typing import List


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.OCR_LLM

    # running a multi-billion parameter VLM on CPU takes hours per page (see
    # omr/steps/text/llm/adapters.py), so this step is GPU-only
    @classmethod
    def default_predictor_resource(cls) -> WorkerResource:
        return WorkerResource.GPU

    @classmethod
    def allowed_predictor_resources(cls) -> List[WorkerResource]:
        return [WorkerResource.GPU]

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import LLMTextPredictor
        return LLMTextPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError("LLM based text transcription has no trainer")

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return TextDataset

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import TextExperimenter
        return TextExperimenter


Step.register(Meta)
