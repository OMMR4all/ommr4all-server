from abc import ABC
from typing import Type, Dict
from .algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, AlgorithmTypes, AlgorithmTrainerSettings
from logging import getLogger


logger = getLogger()


class Step(ABC):
    METAS: Dict[AlgorithmTypes, Type[AlgorithmMeta]] = {
    }

    @staticmethod
    def register(meta: Type[AlgorithmMeta]):
        Step.METAS[meta.type()] = meta

    @staticmethod
    def create_trainer(t: AlgorithmTypes, settings: AlgorithmTrainerSettings) -> AlgorithmTrainer:
        return Step.create_meta(t).create_trainer(settings)

    @staticmethod
    def create_predictor(t: AlgorithmTypes, settings) -> AlgorithmPredictor:
        return Step.create_meta(t).create_predictor(settings)

    @staticmethod
    def trainer(t: AlgorithmTypes) -> Type[AlgorithmTrainer]:
        return Step.create_meta(t).trainer()

    @staticmethod
    def predictor(t: AlgorithmTypes) -> Type[AlgorithmPredictor]:
        return Step.create_meta(t).predictor()

    @staticmethod
    def create_meta(t: AlgorithmTypes) -> Type[AlgorithmMeta]:
        return Step.meta(t)

    @staticmethod
    def meta(t: AlgorithmTypes) -> Type[AlgorithmMeta]:
        try:
            return Step.METAS[t]
        except KeyError as e:
            logger.exception(e)
            raise Exception("Meta {} not found. Add meta by adding missing import!")


# Import all metas which register itself
# * Do not delete
# * New metas must be added here
import omr.steps.preprocessing.meta
import omr.steps.stafflines.detection.pixelclassifier.meta
import omr.steps.layout.lyricsbbs.meta
import omr.steps.layout.simplelyrics.meta
import omr.steps.layout.standard.meta
import omr.steps.symboldetection.pixelclassifier.meta
import omr.steps.symboldetection.sequencetosequence.meta
import omr.steps.symboldetection.torchpixelclassifier.meta
import omr.steps.layout.correction_tools.connectedcomponentsselector.meta
import omr.steps.text.calamari.meta
import omr.steps.syllables.syllablesfromtext.meta
import omr.steps.syllables.syllablesinorder.meta
