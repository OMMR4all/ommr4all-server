import os

from typing import List, Optional, Generator



from omr.confidence.symbol_sequence_confidence import SymbolSequenceConfidenceLookUp, SequenceSetting


from database.file_formats.pcgts import *
from omr.steps.symboldetection.dataset import SymbolDetectionDataset, SymbolDetectionDatasetTorch
from omr.dataset import RegionLineMaskData, DatasetParams
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings, PredictionProgress
import cv2
import numpy as np
from omr.steps.symboldetection.torchpixelclassifier.meta import Meta
from omr.imageoperations.symbol_heads import SYMBOL_DETECTION_HEAD_COUNT, symbol_detection_heads
from omr.imageoperations.symbol_label_set import SymbolClassLabelSets
from omr.steps.symboldetection.predictor import SymbolsPredictor, SingleLinePredictionResult

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
        from segmentation.model_builder import ModelBuilderLoad
        from segmentation.network_postprocessor import NetworkMaskPostProcessor, MaskPredictionResult
        from segmentation.scripts.train import get_default_device
        from segmentation.network import Network, EnsemblePredictor
        path = os.path.join(settings.model.path)
        modelbuilder = ModelBuilderLoad.from_disk(model_weights=os.path.join(path, 'best.torch'),
                                                  device=get_default_device())
        logger.info(f"Using model: {os.path.join(path, 'best.torch')}")

        with open(os.path.join(path, 'dataset_params.json'), 'r') as f:
            self.dataset_params = DatasetParams.from_json(f.read())
        self.label_sets = self.dataset_params.symbol_label_sets or SymbolClassLabelSets.builtin()

        base_model = modelbuilder.get_model()
        config = modelbuilder.get_model_configuration()
        logger.info(config)
        device = get_default_device()
        logger.info(f"Using device: {device}")
        self.predictor = EnsemblePredictor.from_model_config([base_model], [config])
        self.nmaskpredictor = NetworkMaskPostProcessor(self.predictor, config.color_map)
        if len(config.color_map) != len(self.label_sets.main):
            raise ValueError(
                f"Model {path} has {len(config.color_map)} main classes but its "
                f"dataset_params.json declares {len(self.label_sets.main)} symbol labels")
        # match the model's additional heads to the registry by position + class count;
        # unknown heads are ignored instead of crashing on a foreign checkpoint
        n_heads, head_cls = config.head_config()
        self.head_specs = []
        heads = symbol_detection_heads(self.label_sets)
        for i in range(n_heads):
            spec = heads[i] if i < SYMBOL_DETECTION_HEAD_COUNT else None
            if spec is None or len(spec.labels) != head_cls[i]:
                logger.warning(f"Model head {i} ({head_cls[i]} classes) does not match any known "
                               f"symbol head, its predictions are ignored")
                spec = None
            self.head_specs.append(spec)
        self.look_up = SymbolSequenceConfidenceLookUp(SequenceSetting.NOTE_3GRAM)
        # print(self.dataset_params.to_json())

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        from segmentation.network_postprocessor import MaskPredictionResult
        from segmentation.preprocessing.source_image import SourceImage
        dataset = SymbolDetectionDatasetTorch(pcgts_files, self.dataset_params)
        df = dataset.to_memory_dataset(train=False)
        clefs = []
        progress = PredictionProgress.for_lines(callback, pcgts_files, list(df['original']))
        progress.start()
        # Enumerate positionally: df's index label need not match the position in
        # the item -> page map built above.
        for pos, (index, row) in enumerate(df.iterrows()):
            mask, image, data, mask2 = row['masks'], row['images'], row['original'], row.get('add_mask_0')
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
            other_probability_maps = output.prediction_result.other_probability_map or []
            head_softmaxes = [softmax(np.squeeze(pm), axis=-1) if spec is not None else None
                              for pm, spec in zip(other_probability_maps, self.head_specs)]

            m: RegionLineMaskData = data
            symbols = extract_symbols(prob_map_softmax, labels, m, dataset=dataset, min_symbol_area=-1,
                                      clef=self.settings.params.use_rule_based_post_processing and self.settings.params.use_pis_clef_correction , lookup=self.look_up,
                                      probability=0.5, additional_masks=head_softmaxes, heads=self.head_specs,
                                      label_sets=self.label_sets)

            additional_symbols = filter_unique_symbols_by_coord(symbols,
                                                                extract_symbols(prob_map_softmax, labels, m,
                                                                                dataset,
                                                                                probability=0.95,
                                                                                clef=self.settings.params.use_rule_based_post_processing and self.settings.params.use_pis_clef_correction,
                                                                                min_symbol_area=4, lookup=self.look_up,
                                                                                additional_masks=head_softmaxes, heads=self.head_specs,
                                                                                label_sets=self.label_sets))

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


                initial_clef = None
                if len(symbols) > 0:
                    if symbols[0].symbol_type == symbols[0].symbol_type.CLEF:
                        clefs.append(symbols[0])
                        initial_clef = symbols[0]
                    elif len(clefs) > 0:
                        if self.settings.params.use_missing_clef_correction:
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

            progress.item_finished(pos)
            yield single_line_symbols, single_line_symbols_2


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
