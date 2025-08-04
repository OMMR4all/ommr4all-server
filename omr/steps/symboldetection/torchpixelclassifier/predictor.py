import os

from typing import List, Optional, Generator

from omr.confidence.symbol_sequence_confidence import SymbolSequenceConfidenceLookUp, SequenceSetting
from segmentation.model_builder import ModelBuilderLoad
from segmentation.network_postprocessor import NetworkMaskPostProcessor, MaskPredictionResult
from segmentation.scripts.train import get_default_device
from segmentation.preprocessing.source_image import SourceImage

from database.file_formats.pcgts import *
from omr.steps.symboldetection.dataset import SymbolDetectionDataset, SymbolDetectionDatasetTorch
from omr.dataset import RegionLineMaskData, DatasetParams
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings
import cv2
import numpy as np
from omr.steps.symboldetection.torchpixelclassifier.meta import Meta
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.symboldetection.predictor import SymbolsPredictor, SingleLinePredictionResult
from segmentation.network import Network, EnsemblePredictor
from omr.steps.symboldetection.postprocessing.symbol_extraction_from_prob_map import extract_symbols, \
    render_prediction_labels
from omr.steps.symboldetection.postprocessing.symobl_background_knwoledge_postprocessing import *
from loguru import logger

class PCTorchPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        import torch
        use_cuda = torch.cuda.is_available()
        path = os.path.join(settings.model.path)
        modelbuilder = ModelBuilderLoad.from_disk(model_weights=os.path.join(path, 'best.torch'),
                                                  device=get_default_device())
        logger.info(f"Using model: {os.path.join(path, 'best.torch')}")

        # model_path = "/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/modules/ommr4all-server/storage/Graduel_Part_1_gt/models/symbols_pc_torch/2023-02-02T14:01:14/"
        # modelbuilder = ModelBuilderLoad.from_disk(model_weights= model_path + 'best.torch',
        #                                          device=get_default_device())
        with open(os.path.join(path, 'dataset_params.json'), 'r') as f:
            self.dataset_params = DatasetParams.from_json(f.read())

        base_model = modelbuilder.get_model()
        config = modelbuilder.get_model_configuration()
        preprocessing_settings = modelbuilder.get_model_configuration().preprocessing_settings
        self.predictor = EnsemblePredictor([base_model], [preprocessing_settings])
        self.nmaskpredictor = NetworkMaskPostProcessor(self.predictor, config.color_map)
        self.look_up = SymbolSequenceConfidenceLookUp(SequenceSetting.NOTE_3GRAM)
        # print(self.dataset_params.to_json())

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDatasetTorch(pcgts_files, self.dataset_params)
        df = dataset.to_memory_dataset(train=False)
        clefs = []
        total_lines = len(list(df.iterrows()))
        for index, row in df.iterrows():
            mask, image, data, mask2 = row['masks'], row['images'], row['original'], row['add_symbols_mask']
            source_image = SourceImage.from_numpy(image)
            output: MaskPredictionResult = self.nmaskpredictor.predict_image(source_image)
            #output.generated_mask.show()
            #f, ax = plt.subplots(ncols=1,nrows=3, sharex=True, sharey=True)
            #ax[0].imshow(np.array(output.generated_mask))
            #ax[1].imshow(np.transpose(np.squeeze(output.prediction_result.network_input), (1,2,0)) )
            #ax[2].imshow(output.prediction_result.source_image.array())
            #plt.show()

            # output = self.predictor.predict_single_image(image=image)
            labels = np.argmax(output.prediction_result.probability_map, axis=-1)
            from scipy.special import softmax
            prob_map_softmax = softmax(output.prediction_result.probability_map, axis=-1)
            second_mask_softmax = None
            if output.prediction_result.other_probability_map[0] is not None:
                second_mask_softmax = softmax(np.squeeze(output.prediction_result.other_probability_map[0]), axis=-1)

            m: RegionLineMaskData = data
            symbols = extract_symbols(prob_map_softmax, labels, m, dataset=dataset, min_symbol_area=-1,
                                      clef=self.settings.params.use_rule_based_post_processing and self.settings.params.use_rule_based_post_processing , lookup=self.look_up,
                                      probability=0.5, second_mask=second_mask_softmax)

            additional_symbols = filter_unique_symbols_by_coord(symbols,
                                                                extract_symbols(prob_map_softmax, labels, m,
                                                                                dataset,
                                                                                probability=0.95,
                                                                                clef=self.settings.params.use_rule_based_post_processing and self.settings.params.use_pis_clef_correction,
                                                                                min_symbol_area=4, lookup=self.look_up, second_mask=second_mask_softmax))

            if self.settings.params.use_rule_based_post_processing:
                if self.settings.params.use_block_layout_correction:
                    symbols = correct_symbols_inside_wrong_blocks(m.operation.page, symbols)
                    symbols = correct_symbols_inside_text_blocks(m.operation.page, symbols)

                if self.settings.params.use_overlapping_symbol_correction:
                    symbols = fix_overlapping_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2)

                additional_symbols = correct_symbols_inside_text_blocks(m.operation.page, additional_symbols)
                #additional_symbols = correct_symbols_inside_wrong_blocks(m.operation.page, additional_symbols)
                additional_symbols = correct_symbols_inside_text_blocks(m.operation.page, additional_symbols)

                if self.settings.params.use_missing_clef_correction:
                    symbols, change = fix_missing_clef(symbols, additional_symbols)
                    symbols = fix_missing_clef2(symbols1=symbols, symbols2=additional_symbols, page=m.operation.page, m=m)

                #symbols = fix_overlapping_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2)
                ### symbols = fix_pos_of_close_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m)
                if self.settings.params.use_graphical_connection_correction:
                    correct_looped_connection(symbols, additional_symbols, page=m.operation.page, m=m)

                if self.settings.params.use_pis_correction_of_stacked_symbols:
                    symbols = fix_pos_of_close_symbols3(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m)

                symbols = add_neume_start_pos(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m, debug=False)
                line = Line(symbols=symbols)

                if self.settings.params.use_missing_clef_correction:

                    initial_clef = None
                    if len(symbols) > 0:
                        if symbols[0].symbol_type == symbols[0].symbol_type.CLEF:
                            clefs.append(symbols[0])
                            initial_clef = symbols[0]
                        elif len(clefs) > 0:
                            initial_clef = clefs[-1]
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
                                                             data)
            single_line_symbols_2 = SingleLinePredictionResult(additional_symbols,
                                                               data)
            if False:
                from shared.pcgtscanvas import PcGtsCanvas
                canvas = PcGtsCanvas(m.operation.page, PageScaleReference.NORMALIZED_X2)
                for s in symbols.symbols:
                    s.coord = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(s.coord.x,
                                                                                                    s.position_in_staff)
                canvas.draw(symbols.symbols, invert=True)
                canvas.show()
            if False:
                labels2 = np.argmax(np.squeeze(output.prediction_result.other_probability_map[0]), axis=-1)

                import matplotlib.pyplot as plt
                f, ax = plt.subplots(6, 1, sharey='all', sharex='all')
                ax[0].imshow(output.prediction_result.probability_map[:, :, 0])  # , vmin=0.0, vmax=1.0)
                ax[1].imshow(image, vmin=0.0, vmax=255)
                ax[2].imshow(render_prediction_labels(mask, image))
                ax[3].imshow(render_prediction_labels(mask2, image))
                ax[4].imshow(render_prediction_labels(labels, image))
                ax[5].imshow(render_prediction_labels(labels2, image))
                plt.show()

            if callback:
                percentage = (index + 1) / total_lines

                callback.progress_updated(percentage, n_processed_pages=index + 1, n_pages=total_lines)
            yield single_line_symbols, single_line_symbols_2

    def extract_symbols123(self, probs: np.ndarray, p: np.ndarray, m: RegionLineMaskData,
                           dataset: SymbolDetectionDataset) -> List[MusicSymbol]:
        # n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(((probs[:, :, 0] < 0.5) | (p > 0)).astype(np.uint8))
        p = (np.argmax(probs[:, :, 1:], axis=-1) + 1) * (probs[:, :, 0] < 0.5)
        n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8))
        symbols = []
        sorted_labels = sorted(range(1, n_labels), key=lambda i: (centroids[i, 0], -centroids[i, 1]))
        centroids_canvas = np.zeros(p.shape, dtype=np.uint8)
        for i in sorted_labels:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            # if a <= 4:
            #    continue
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]
            c = Point(x=centroids[i, 0], y=centroids[i, 1])
            coord = dataset.local_to_global_pos(c, m.operation.params)
            coord = m.operation.page.image_to_page_scale(coord, m.operation.scale_reference)
            # coord = coord.round().astype(int)
            # compute label this the label with the hightest frequency of the connected component
            area = p[y:y + h, x:x + w] * (cc[y:y + h, x:x + w] == i)
            label = SymbolLabel(int(np.argmax([np.sum(area == v + 1) for v in range(len(SymbolLabel) - 1)])) + 1)
            centroids_canvas[int(np.round(c.y)), int(np.round(c.x))] = label
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord)
            if label == SymbolLabel.NOTE_START:
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.NEUME_START,
                ))
            elif label == SymbolLabel.NOTE_GAPPED:
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.GAPED,
                ))
            elif label == SymbolLabel.NOTE_LOOPED:
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.LOOPED,
                ))
            elif label == SymbolLabel.CLEF_C:
                symbols.append(create_clef(ClefType.C, coord=coord, position_in_staff=position_in_staff))
            elif label == SymbolLabel.CLEF_F:
                symbols.append(create_clef(ClefType.F, coord=coord, position_in_staff=position_in_staff))
            elif label == SymbolLabel.ACCID_FLAT:
                symbols.append(create_accid(AccidType.FLAT, coord=coord))
            elif label == SymbolLabel.ACCID_SHARP:
                symbols.append(create_accid(AccidType.SHARP, coord=coord))
            elif label == SymbolLabel.ACCID_NATURAL:
                symbols.append(create_accid(AccidType.NATURAL, coord=coord))
            else:
                raise Exception("Unknown label {} during decoding".format(label))

        if False:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(6, 1, sharex='all', sharey='all')
            ax[0].imshow(p)
            ax[1].imshow(m.mask)
            ax[2].imshow(render_prediction_labels(centroids_canvas, m.region))
            ax[2].imshow(render_prediction_labels(centroids_canvas, m.region))

            labels = render_prediction_labels(p, 255 - m.region)
            ax[3].imshow(labels)
            ax[4].imshow(m.region, cmap='gray_r')
            ax[5].imshow(cc, cmap='gist_ncar_r')
            plt.show()

        return symbols


if __name__ == '__main__':
    if __name__ == '__main__':
        import django

        os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
        django.setup()

    from database import DatabaseBook

    b = DatabaseBook('mulhouse_mass_transcription')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()][7:20]
    pred = PCTorchPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
    import matplotlib.pyplot as plt

    for i in ps:
        if len(i.music_lines) > 0:
            orig = np.array(i.music_lines[0].line.operation.page_image)
            for p in i.music_lines:
                for s in p.symbols:
                    if s.symbol_type == SymbolType.NOTE or SymbolType.CLEF:
                        c = p.line.operation.page.page_to_image_scale(s.coord,
                                                                      ref=PageScaleReference.NORMALIZED_X2).round().astype(
                            int)
                        t, l = c.y, c.x
                        orig[t - 2:t + 2, l - 2:l + 2] = 255

            plt.imshow(orig)
            plt.show()
        else:
            print("no")
