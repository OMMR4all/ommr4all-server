from abc import ABC
from typing import Type, Dict
from .algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, AlgorithmTypes, AlgorithmTrainerSettings
from logging import getLogger


logger = getLogger()


class Step(ABC):
    METAS: Dict[AlgorithmTypes, Type[AlgorithmMeta]] = {
    }
    _INITIALIZED = False

    @staticmethod
    def register(meta: Type[AlgorithmMeta]):
        Step.METAS[meta.type()] = meta

    @classmethod
    def _lazy_load_registry(cls):
        logger.info("OMR Lazy Register: Check")

        if cls._INITIALIZED:
            return

        logger.info("OMR RESTRY: Lazy loading algorithm metas...")
        try:
            # Import all metas which register itself
            # * Do not delete
            # * New metas must be added here
            import omr.steps.preprocessing.meta
            import omr.steps.stafflines.detection.pixelclassifier.meta
            import omr.steps.stafflines.detection.pixelclassifier_torch.meta
            import omr.steps.layout.lyricsbbs.meta
            import omr.steps.layout.simplelyrics.meta
            import omr.steps.layout.drop_capitals_yolo.meta

            import omr.steps.layout.standard.meta
            # import omr.steps.symboldetection.pixelclassifier.meta
            import omr.steps.symboldetection.torchpixelclassifier.meta
            # import omr.steps.symboldetection.sequencetosequence.meta
            # import omr.steps.symboldetection.sequence_to_sequence_nautilus.meta
            import omr.steps.symboldetection.sequence_to_sequence_guppy.meta
            import omr.steps.symboldetection.yolo_detector.meta

            import omr.steps.postprocessing.meta

            import omr.steps.layout.correction_tools.connectedcomponentsselector.meta
            # import omr.steps.text.calamari.meta
            # import omr.steps.text.pytorch_ocr.meta
            import omr.steps.text.guppy.meta

            # import omr.steps.syllables.syllablesfromtext.meta
            import omr.steps.syllables.syllablesfromtexttorch.meta

            import omr.steps.syllables.syllablesinorder.meta

            import omr.steps.text.correction_tools.document_matching_corrector.meta
            import omr.steps.text.correction_tools.document_corrector.meta
            import omr.steps.tools.meta
            cls._INITIALIZED = True
            logger.debug(f"OMR REGISTRY: Successfully registered {len(cls.METAS)} algorithms.")

        except Exception as e:
            logger.exception("OMR REGISTRY: Failed to load algorithm metas")
            raise e
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
        Step._lazy_load_registry()
        try:
            return Step.METAS[t]
        except KeyError:
            logger.error(f"Meta {t} not found. Ensure the import is in _lazy_load_registry.")
            raise Exception(f"Algorithm Meta {t} not found.")

