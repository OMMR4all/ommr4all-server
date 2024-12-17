import os

from typing import List, Optional, Generator

from omr.confidence.symbol_sequence_confidence import SymbolSequenceConfidenceLookUp, SequenceSetting
from segmentation.model_builder import ModelBuilderLoad
from segmentation.network_postprocessor import NetworkMaskPostProcessor
from segmentation.scripts.train import get_default_device
from segmentation.preprocessing.source_image import SourceImage

from database.file_formats.pcgts import *
from omr.steps.symboldetection.dataset import SymbolDetectionDataset, SymbolDetectionDatasetTorch
from omr.dataset import RegionLineMaskData, DatasetParams
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings
import cv2
import numpy as np
from omr.steps.symboldetection.yolo_detector.meta import Meta
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.symboldetection.predictor import SymbolsPredictor, SingleLinePredictionResult
from segmentation.network import Network, EnsemblePredictor
from omr.steps.symboldetection.postprocessing.symbol_extraction_from_prob_map import extract_symbols
from omr.steps.symboldetection.postprocessing.symobl_background_knwoledge_postprocessing import *
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class PCTorchPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        import torch
        use_cuda = torch.cuda.is_available()
        path = os.path.join(settings.model.path)

        #modelbuilder = ModelBuilderLoad.from_disk(model_weights=os.path.join(path, 'best.torch'),
        #                                          device=get_default_device())
        #logger.info(f"Using model: {os.path.join(path, 'best.torch')}")
        #path = "/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/runs/detect/yolov11n.pt"
        #olds = "/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/runs/detect/yolov8n.pt5/weights/best.pt"
        #self.model = YOLO("/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/runs/detect/yolov11n.pt/weights/best.pt")
        self.model = YOLO(os.path.join(path, "yolo11n", "weights", "best.pt"))
        # model_path = "/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/modules/ommr4all-server/storage/Graduel_Part_1_gt/models/symbols_pc_torch/2023-02-02T14:01:14/"
        # modelbuilder = ModelBuilderLoad.from_disk(model_weights= model_path + 'best.torch',
        #                                          device=get_default_device())
        path = os.path.join(settings.model.path)

        with open(os.path.join(path, 'dataset_params.json'), 'r') as f:
            self.dataset_params = DatasetParams.from_json(f.read())

        #base_model = modelbuilder.get_model()
        #config = modelbuilder.get_model_configuration()
        #preprocessing_settings = modelbuilder.get_model_configuration().preprocessing_settings
        #self.predictor = EnsemblePredictor([base_model], [preprocessing_settings])
        #self.nmaskpredictor = NetworkMaskPostProcessor(self.predictor, config.color_map)
        self.look_up = SymbolSequenceConfidenceLookUp(SequenceSetting.NOTE_3GRAM)
        # print(self.dataset_params.to_json())

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDatasetTorch(pcgts_files, self.dataset_params)
        df = dataset.to_memory_dataset(train=False)
        clefs = []

        for index, row in df.iterrows():
            mask, image, data = row['masks'], row['images'], row['original']
            image2 = image[:, :, [2, 1, 0]]

            # from matplotlib import pyplot as plt
            #plt.imshow(image)
            #plt.show()
            #plt.imshow(mask)
            #plt.show()
            #source_image = SourceImage.from_numpy(image)
            #output: MaskPredictionResult = self.nmaskpredictor.predict_image(source_image)
            #output.generated_mask.show()
            #f, ax = plt.subplots(ncols=1,nrows=3, sharex=True, sharey=True)
            #ax[0].imshow(np.array(output.generated_mask))
            #ax[1].imshow(np.transpose(np.squeeze(output.prediction_result.network_input), (1,2,0)) )
            #ax[2].imshow(output.prediction_result.source_image.array())
            #plt.show()
            output = self.model.predict(image2)
            #print(output)
            #single_line_symbols = SingleLinePredictionResult(symbols, data)
            symbols = []
            for r in output:
                #annotator = Annotator(image)
                boxes = r.boxes
                for box in boxes:
                    #print(box)
                    b = box.xyxy[0]
                    x1 = b[0]
                    y1 = b[1]
                    x2 = b[2]
                    y2 = b[3]
                    width = x2 -x1
                    height = y2 -y1
                    cx = int(x1 + 0.5 * width)
                    cy = int(y1 + 0.5 * height)

                    c = box.cls
                    #print(f"la{c[0]} center: cx{cx} cy{cy}")
                    label = SymbolLabel(c[0].cpu().numpy() + 1)
                    coord = Point(cx, cy)
                    coord = dataset.local_to_global_pos(coord, data.operation.params)
                    coord = data.operation.page.image_to_page_scale(coord, data.operation.scale_reference)
                    # coord = coord.round().astype(int)
                    # compute label this the label with the hightest frequency of the connected component

                    position_in_staff = data.operation.music_line.compute_position_in_staff(coord)
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

                    #annotator.box_label(b, "")
                #img = annotator.result()
                #from matplotlib import pyplot as plt
                #plt.imshow(img)
                #plt.show()
            yield SingleLinePredictionResult(symbols, data)

if __name__ == '__main__':
    if __name__ == '__main__':
        import django

        os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
        django.setup()

    from database import DatabaseBook

    b = DatabaseBook('Graduel_Part_1')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()][7:20]
    pred = PCTorchPredictor(AlgorithmPredictorSettings(Meta.default_model_for_style("french14")))
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
