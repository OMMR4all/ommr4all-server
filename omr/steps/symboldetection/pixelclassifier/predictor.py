import os
import random

import shapely.geometry
from PIL import Image
from shapely.geometry import Polygon

from database.file_formats.pcgts.page import SymbolErrorType
from omr.confidence.symbol_sequence_confidence import SymbolSequenceConfidenceLookUp, SequenceSetting

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import List, Optional, Generator
from ocr4all_pixel_classifier.lib.predictor import Predictor, PredictSettings
from database.file_formats.pcgts import *
from omr.steps.symboldetection.dataset import SymbolDetectionDataset
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings
import cv2
import numpy as np
from omr.steps.symboldetection.pixelclassifier.meta import Meta
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.symboldetection.predictor import SymbolsPredictor, SingleLinePredictionResult


from ocr4all_pixel_classifier.lib.model import Architecture
from ocr4all_pixel_classifier.lib.network import Network
from omr.steps.symboldetection.postprocessing.symbol_extraction_from_prob_map import extract_symbols
from omr.steps.symboldetection.postprocessing.symobl_background_knwoledge_postprocessing import *




class PCPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        settings = PredictSettings(
            n_classes=len(SymbolLabel),
            network=os.path.join(settings.model.local_file('model.h5'))
        )
        self.predictor = Predictor(settings)
        self.look_up = SymbolSequenceConfidenceLookUp(SequenceSetting.NOTE_3GRAM)

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDataset(pcgts_files, self.dataset_params)
        clefs = []
        for p in self.predictor.predict(dataset.to_page_segmentation_dataset()):
            m: RegionLineMaskData = p.data.user_data
            symbols = extract_symbols(p.probabilities, p.labels, m, dataset,
                                           clef=self.settings.params.use_clef_pos_correction, lookup= self.look_up)
            additional_symbols = filter_unique_symbols_by_coord(symbols,
                                                                extract_symbols(p.probabilities, p.labels, m,
                                                                                     dataset,
                                                                                     probability=0.95,
                                                                                     clef=self.settings.params.use_clef_pos_correction, lookup= self.look_up))
            if True:
                #symbols = correct_symbols_inside_wrong_blocks(m.operation.page, symbols)
                #symbols = correct_symbols_inside_text_blocks(m.operation.page, symbols)
                symbols = fix_overlapping_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2)

                additional_symbols = correct_symbols_inside_text_blocks(m.operation.page, additional_symbols)
                additional_symbols = correct_symbols_inside_wrong_blocks(m.operation.page, additional_symbols)
                additional_symbols = correct_symbols_inside_text_blocks(m.operation.page, additional_symbols)

                symbols, change = fix_missing_clef(symbols, additional_symbols)
                symbols = fix_missing_clef2(symbols1=symbols, symbols2=additional_symbols, page=m.operation.page, m=m)
                symbols = fix_overlapping_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2)
                # symbols = fix_pos_of_close_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m)
                correct_looped_connection(symbols, additional_symbols, page=m.operation.page, m=m)
                symbols = fix_pos_of_close_symbols3(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m)

                initial_clef = None
                if len(symbols) > 0:
                    if symbols[0].symbol_type == symbols[0].symbol_type.CLEF:
                        clefs.append(symbols[0])
                        initial_clef = symbols[0]
                    elif len(clefs) > 0:
                        # symbols.insert(0, clefs[-1])
                        initial_clef = clefs[-1]
                line = Line(symbols=symbols)
                line.update_note_names(initial_clef=initial_clef)
                symbols = line.symbols
                '''
                if len(symbols) > 0:
                    if symbols[0].symbol_type != symbols[0].symbol_type.CLEF:
                        print(symbols[0].symbol_type)
                        if len(additional_symbols) > 0:
                            print(additional_symbols[0].symbol_type)
                        print(m.operation.page.location.page)
                '''
            single_line_symbols = SingleLinePredictionResult(symbols,
                                                             p.data.user_data)
            single_line_symbols_2 = SingleLinePredictionResult(additional_symbols,
                                                               p.data.user_data)

            if False:
                from shared.pcgtscanvas import PcGtsCanvas
                canvas = PcGtsCanvas(m.operation.page, PageScaleReference.NORMALIZED_X2)
                # for s in single_line_symbols.symbols:
                #    s.coord = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(s.coord.x,
                #                                                                                    s.position_in_staff)
                canvas.draw(single_line_symbols.symbols, invert=True)
                canvas.show()
            if False:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots(6, 1, sharey='all', sharex='all')
                ax[0].imshow(p.probabilities[:, :, 0], vmin=0.0, vmax=1.0)
                ax[1].imshow(p.data.image, vmin=0.0, vmax=255)
                ax[2].imshow(render_prediction_labels(p.labels, p.data.image))
                ax[3].imshow((p.probabilities[:, :, 0] <= 0.8) * (1 + np.argmax(p.probabilities[:, :, 1:], axis=-1)))
                ax[4].imshow(render_prediction_labels(p.data.mask, p.data.image))
                ax[5].imshow(render_prediction_labels(p.data.image, p.data.image))

                plt.show()

            yield single_line_symbols, single_line_symbols_2




if __name__ == '__main__':
    from database import DatabaseBook

    b = DatabaseBook('Pa_14819')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[0:1]]

    pred = PCPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
    import matplotlib.pyplot as plt

    orig = np.array(ps[0].music_lines[0].line.operation.page_image)
    for p in ps[0].music_lines:
        for s in p.symbols:
            if s.symbol_type == SymbolType.NOTE:
                c = p.line.operation.page.page_to_image_scale(s.coord,
                                                              ref=PageScaleReference.NORMALIZED_X2).round().astype(int)
                t, l = c.y, c.x
                orig[t - 2:t + 2, l - 2:l + 2] = 255

    plt.imshow(orig)
    plt.show()
