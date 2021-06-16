import os

from database.file_formats.performance.pageprogress import Locks

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import List, Optional, Generator
from centernet.predict import predict
from database.file_formats.pcgts import *
from omr.steps.symboldetection.dataset import SymbolDetectionDataset
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings
import cv2
import numpy as np
from omr.steps.symboldetection.centernet.meta import Meta
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.symboldetection.predictor import SymbolsPredictor, SingleLinePredictionResult
import pandas as pd

def render_prediction_labels(labels, img=None):
    from shared.pcgtscanvas import PcGtsCanvas
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    if img is not None:
        out = np.stack((img,) * 3, axis=-1).astype(int)

    def draw(i, c):
        return np.kron((labels == i), c).reshape(out.shape).astype(int)

    for i, c in [
        (SymbolLabel.BACKGROUND, (255, 255, 255)),
        (SymbolLabel.NOTE_START, (255, 0, 0)),
        (SymbolLabel.NOTE_GAPPED, (255, 120, 120)),
        (SymbolLabel.NOTE_LOOPED, (120, 0, 0)),
        (SymbolLabel.CLEF_C, (120, 255, 120)),
        (SymbolLabel.CLEF_F, (0, 255, 0)),
        (SymbolLabel.ACCID_NATURAL, (0, 0, 255)),
        (SymbolLabel.ACCID_SHARP, (50, 50, 255)),
        (SymbolLabel.ACCID_FLAT, (0, 0, 120)),
    ]:
        c = PcGtsCanvas.color_for_music_symbol(i.to_music_symbol(), inverted=True, default_color=(255, 255, 255))
        if c != (0, 0, 0):
            out[:, :, 0] = np.where(labels == i, c[0], out[:, :, 0])
            out[:, :, 1] = np.where(labels == i, c[1], out[:, :, 1])
            out[:, :, 2] = np.where(labels == i, c[2], out[:, :, 2])

    # if img is not None:
    # out = (out.astype(float) * np.stack((img,) * 3, axis=-1) / 255).astype(np.uint8)

    return out.clip(0, 255).astype(np.uint8)

class Config:
    def as_dict(self):
        return vars(self)

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self)

class CenterNetPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings = None):
        super().__init__(settings)
        config = Config()
        config.train_dir = 'train_images'
        config.test_dir = 'test_images'
        config.batch_size = 1
        config.fold = 0
        config.num_folds = 10
        config.device = 'cuda'
        config.p_letter = 0.2
        config.p_class = 0.2
        config.tta = True
        config.scales = [0.5, 1.0]
        config.train_csv = 'train.csv'
        config.slug = 'r50'
        config.w_h_ratio = 0.2
        config.pred_zip = None

        config.weight = '/home/alexander/PycharmProjects/ommr4all-server/omr/steps/symboldetection/centernet/f01-ep-0040-val_hm_acc-0.9660-val_classes_acc-0.2354.pth'
        self.config = config
        #settings = PredictSettings(
        #    n_classes=len(SymbolLabel),
        #    network=os.path.join(settings.model.local_file('model.h5'))
        #)
       # self.predictor = Predictor(settings)

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDataset(pcgts_files, self.dataset_params)
        test_data = dataset.to_centernet_dataset()
        images, bboxes = test_data
        ratio = int(0.8 * len(bboxes))
        df_train = pd.DataFrame(data={'images': images[:ratio], 'bbox': bboxes[:ratio]})
        df_val = pd.DataFrame(data={'images': images[:ratio], 'bbox': bboxes[:ratio]})
        predict(df_val, self.config)

        '''
        for p in self.predictor.predict(dataset.to_page_segmentation_dataset()):
            m: RegionLineMaskData = p.data.user_data
            symbols = SingleLinePredictionResult(self.exract_symbols(p.probabilities, p.labels, m, dataset),
                                                 p.data.user_data)
            if False:
                from shared.pcgtscanvas import PcGtsCanvas
                canvas = PcGtsCanvas(m.operation.page, PageScaleReference.NORMALIZED_X2)
                for s in symbols.symbols:
                    s.coord = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(s.coord.x,
                                                                                                    s.position_in_staff)
                canvas.draw(symbols.symbols, invert=True)
                canvas.show()
            if False:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots(5, 1, sharey='all', sharex='all')
                ax[0].imshow(p.probabilities[:, :, 0], vmin=0.0, vmax=1.0)
                ax[1].imshow(p.data.image, vmin=0.0, vmax=255)
                ax[2].imshow(render_prediction_labels(p.labels, p.data.image))
                ax[3].imshow((p.probabilities[:, :, 0] <= 0.8) * (1 + np.argmax(p.probabilities[:, :, 1:], axis=-1)))
                ax[4].imshow(render_prediction_labels(p.data.mask, p.data.image))
                plt.show()
            '''
            #yield symbols

    def exract_symbols(self, probs: np.ndarray, p: np.ndarray, m: RegionLineMaskData,
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
            if a <= 4:
                continue
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
            f, ax = plt.subplots(5, 1, sharex='all', sharey='all')
            ax[0].imshow(p)
            ax[1].imshow(render_prediction_labels(centroids_canvas, m.region))
            labels = render_prediction_labels(p, 255 - m.region)
            ax[2].imshow(labels)
            ax[3].imshow(m.region, cmap='gray_r')
            ax[4].imshow(cc, cmap='gist_ncar_r')
            plt.show()

        return symbols


if __name__ == '__main__':
    from database import DatabaseBook
    from omr.dataset import DatasetParams

    b = DatabaseBook('Pa_14819')#Graduel_Part_1')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[3:4]]
    pred = CenterNetPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
    import matplotlib.pyplot as plt
    '''
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
    '''