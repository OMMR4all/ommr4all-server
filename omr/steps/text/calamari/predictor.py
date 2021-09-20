import os

from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline
from calamari_ocr.ocr.predict.params import PredictorParams
from dataclasses import field, dataclass

#from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalization, LyricsNormalizationParams
from tfaip.data.databaseparams import DataPipelineParams

from omr.steps.text.calamari.calamari_interface import RawData

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from calamari_ocr.ocr.predict.predictor import Predictor, MultiPredictor
from calamari_ocr.ocr.voting import voter_from_params
from calamari_ocr.ocr.voting.params import VoterParams, VoterType
from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import CTCDecoderParams

from typing import List, Tuple, Type, Optional, Generator
from ommr4all.settings import BASE_DIR
from copy import deepcopy

from database.model import Model
from omr.dataset.datastructs import RegionLineMaskData
from database import DatabaseBook
from omr.steps.algorithm import AlgorithmMeta, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.dataset import TextDataset
from omr.steps.text.predictor import \
    TextPredictor, \
    PredictionResult, Point, SingleLinePredictionResult
import numpy as np
from calamari_ocr.utils import glob_all
from database.model.definitions import MetaId
from omr.steps.text.hyphenation.hyphenator import HyphenatorFromDictionary, Pyphenator


from typing import Generator

from tfaip.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.datareader.base import SampleMeta, InputSample
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGenerator


class CalamariPredictor(TextPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.calamari.meta import Meta
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
       # ctc_decoder_params = deepcopy(settings.params.ctcDecoder.params)
       # lnp = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.ONE_STRING))
       # if len(ctc_decoder_params.dictionary) > 0:
       #     ctc_decoder_params.dictionary[:] = [lnp.apply(word) for word in ctc_decoder_params.dictionary]
       # else:
       #     with open(os.path.join(BASE_DIR, 'internal_storage', 'resources', 'hyphen_dictionary.txt')) as f:
       #         # TODO: dataset params in settings, that we can create the correct normalization params
       #         ctc_decoder_params.dictionary[:] = [lnp.apply(line.split()[0]) for line in f.readlines()]

        # self.predictor = MultiPredictor(glob_all([s + '/text_best*.ckpt.json' for s in params.checkpoints]))
        voter_params = VoterParams()
        voter_params.type = VoterParams.type.ConfidenceVoterDefaultCTC
        self.predictor = MultiPredictor.from_paths(
            checkpoints=glob_all([settings.model.local_file('text_best_*.ckpt.json')]),
            voter_params=voter_params,
            predictor_params=PredictorParams(silent=True,
                                             progress_bar=True,
                                             pipeline=DataPipelineParams(batch_size=1,
                                                                         mode=PipelineMode("prediction"))
                                             )
        )
        # self.height = self.predictor.predictors[0].network_params.features
        print(glob_all([settings.model.local_file('text_best_0.ckpt.json')]))
        self.voter = voter_from_params(voter_params)
        #self.predictor = MultiPredictor(glob_all([settings.model.local_file('text_best.ckpt.json')]),
        #                                ctc_decoder_params=ctc_decoder_params)
        #self.height = self.predictor.predictors[0].network_params.features
        #voter_params = VoterParams()
        #voter_params.type = VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC
        #self.voter = voter_from_proto(voter_params)

    def _predict(self, dataset: TextDataset, callback: Optional[PredictionCallback] = None) -> Generator[SingleLinePredictionResult, None, None]:
        hyphen = Pyphenator()
        """
        hyphen = HyphenatorFromDictionary(
            dictionary=os.path.join(BASE_DIR, 'internal_storage', 'resources', 'hyphen_dictionary.txt'),
            normalization=dataset.params.lyrics_normalization,
        )
        """
        #predict_params = PipelineParams(
        #    type=DataSetType.RAW,
        #    skip_invalid=True,
        #    remove_invalid=True,
        #    files=[],
        #    text_files=[],
        #    data_reader_args=FileDataReaderArgs(
        #        pad=None,
        #        text_index=0,
        #    ),
        #    batch_size=1,
        #    num_processes=1,
        #)
        #pipeline: CalamariPipeline = self.predictor.data.get_predict_data(predict_params)
        dataset_cal = dataset.to_text_line_calamari_dataset()

        data_params = dataset_cal
        #data: CalamariDataGeneratorParams = field(
        #    default_factory=FileDataParams,
        #    metadata=pai_meta(mode="flat", choices=DATA_GENERATOR_CHOICES),
        #)
        predictor = self.predictor.predict(data_params)
        pipeline: CalamariPipeline = self.predictor.data.get_or_create_pipeline(self.predictor.params.pipeline, data_params)
        print(predictor)
        #RawDataPipeline
        dataset_cal = dataset.to_text_line_calamari_dataset()
        #pipeline._reader = dataset_cal
        reader = pipeline.reader()
        #pipeline: CalamariPipeline = predictor.data.get_or_create_pipeline(predictor.params.pipeline, args.data)
        #pipeline.
        # reader = pipeline.reader()
        if len(reader) == 0:
            raise Exception("Empty dataset provided. Check your files argument (got {})!".format(len(dataset.files)))

        avg_sentence_confidence = 0
        n_predictions = 0
        for m, s in zip(dataset.load(), predictor):
            inputs, (result, prediction), meta = s.inputs, s.outputs, s.meta
            sample = reader.sample_by_id(meta['id'])
            n_predictions += 1
            sentence = prediction.sentence
            hyphenated = hyphen.apply_to_sentence(prediction.sentence)

            avg_sentence_confidence += prediction.avg_char_probability
            yield SingleLinePredictionResult(self.extract_symbols(dataset, prediction, m), m, hyphenated=hyphenated)
        #try:
        #    for marked_symbols, (r, sample) in zip(dataset.load(), self.predictor.predict_dataset(dataset.to_text_line_calamari_dataset())):
        #        prediction = self.voter.vote_prediction_result(r)
        #        hyphenated = hyphen.apply_to_sentence(prediction.sentence)
        #        yield SingleLinePredictionResult(self.extract_symbols(dataset, prediction, marked_symbols), marked_symbols, hyphenated)
        #except Exception as e:
        #    if str(e) == 'Empty data set provided.':
        #        return
        #
        #    raise e

    def extract_symbols(self, dataset: TextDataset, p, m: RegionLineMaskData) -> List[Tuple[str, Point]]:
        def i2p(p):
            return m.operation.page.image_to_page_scale(p, m.operation.scale_reference)

        if len(p.sentence) > 0 and len(p.positions) == 0:
            # TODO: remove if lm decoding returns positions
            return [(c, None) for c in p.sentence]

        sentence = [(pos.chars[0].char,
                     i2p(dataset.local_to_global_pos(Point((pos.global_start + pos.global_end) / 2, m.operation.text_line.aabb.bottom()), m.operation.params).x))
                    for pos in p.positions]
        return sentence


if __name__ == '__main__':
    from omr.steps.step import Step, AlgorithmTypes
    from ommr4all.settings import BASE_DIR
    import random
    import cv2
    import matplotlib.pyplot as plt
    from shared.pcgtscanvas import PcGtsCanvas
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.LAYOUT, True)], True, [
            # DatabaseBook('Graduel_Part_1'),
            # DatabaseBook('Graduel_Part_2'),
            # DatabaseBook('Graduel_Part_3'),
        ])
    book = DatabaseBook('Gothic_Test')
    meta = Step.meta(AlgorithmTypes.OCR_CALAMARI)
    # model = meta.newest_model_for_book(book)
    model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical', meta.type()))
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict(book.pages()[0:1]))
    for i, p in enumerate(ps):
        canvas = PcGtsCanvas(p.pcgts.page, p.text_lines[0].line.operation.scale_reference)
        for j, s in enumerate(p.text_lines):
            canvas.draw(s)

        canvas.show()

