from typing import List, NamedTuple, Generator
from thirdparty.page_segmentation.lib.predictor import Predictor, PredictSettings
import os
from omr.datatypes import PcGts, Symbol, SymbolType, NoteComponent, Neume, Point, GraphicalConnectionType, \
    Clef, ClefType, AccidentalType, Accidental
from omr.dataset.pcgtsdataset import PcGtsDataset, MusicLineAndMarkedSymbol
import cv2
import numpy as np
from omr.imageoperations.music_line_operations import SymbolLabel

class PredictionResult(NamedTuple):
    symbols: List[Symbol]
    line: MusicLineAndMarkedSymbol


class PCPredictor:
    def __init__(self, checkpoints: List[str]):
        self.height = 80
        settings = PredictSettings(
            network=os.path.splitext(checkpoints[0])[0]
        )
        self.predictor = Predictor(settings)
        self.dataset: PcGtsDataset = None

    def predict(self, pcgts_files: List[PcGts]) -> Generator[PredictionResult, None, None]:
        self.dataset = PcGtsDataset(pcgts_files, gt_required=False, height=self.height)
        for p in self.predictor.predict(self.dataset.to_music_line_page_segmentation_dataset()):
            if False:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots(5, 1, sharey='all', sharex='all')
                ax[0].imshow(p.probabilities[:,:,0])
                ax[1].imshow(p.data.image)
                ax[2].imshow(p.labels)
                ax[3].imshow((p.probabilities[:, :, 0] <= 0.8) * (1 + np.argmax(p.probabilities[:, :, 1:], axis=-1)))
                ax[4].imshow(p.data.mask)
                plt.show()
            m: MusicLineAndMarkedSymbol = p.data.user_data
            yield PredictionResult(self.exract_symbols(p.labels, m), p.data.user_data)

    def exract_symbols(self, p: np.ndarray, m: MusicLineAndMarkedSymbol) -> List[Symbol]:
        n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8))
        symbols = []
        sorted_labels = sorted(range(n_labels), key=lambda i: (centroids[i, 0], -centroids[i, 1]))
        for i in sorted_labels:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]
            c = Point(x=centroids[i, 0], y=centroids[i, 1]).astype(int)
            coord = self.dataset.line_and_mask_operations.local_to_global_pos(c, m.operation.params).astype(int)

            # compute label this the label with the hightest frequency of the connected component
            area = p[y:y+h, x:x+w] * (cc[y:y+h, x:x+w] == i)
            label = SymbolLabel(int(np.argmax([np.sum(area == v + 1) for v in range(len(SymbolLabel) - 1)])) + 1)
            if label == SymbolLabel.NOTE_START:
                symbols.append(Neume(notes=[NoteComponent(coord=coord)]))
            elif label == SymbolLabel.NOTE_GAPPED or label == SymbolLabel.NOTE_LOOPED:
                if len(symbols) > 0 and isinstance(symbols[-1], Neume):
                    n: Neume = symbols[-1]
                    if label == SymbolLabel.NOTE_GAPPED:
                        n.notes.append(NoteComponent(coord=coord, graphical_connection=GraphicalConnectionType.GAPED))
                    else:
                        n.notes.append(NoteComponent(coord=coord, graphical_connection=GraphicalConnectionType.LOOPED))
                else:
                    symbols.append(Neume(notes=[NoteComponent(coord=coord)]))

            elif label == SymbolLabel.CLEF_C:
                symbols.append(Clef(clef_type=ClefType.CLEF_C, coord=coord))
            elif label == SymbolLabel.CLEF_F:
                symbols.append(Clef(clef_type=ClefType.CLEF_F, coord=coord))



        return symbols


if __name__ == '__main__':
    import main.book as book
    b = book.Book('demo')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[0:1]]
    pred = PCPredictor([b.local_path(os.path.join('pc_paths', 'model'))])
    ps = list(pred.predict(val_pcgts))
    import matplotlib.pyplot as plt
    orig = np.array(ps[0].line.operation.page_image)
    for p in ps:
        for s in p.symbols:
            if s.symbol_type == SymbolType.NEUME:
                n: Neume = s
                for nc in n.notes:
                    t, l = nc.coord.y, nc.coord.x
                    orig[t - 2:t + 2, l-2:l+2] = 255

    plt.imshow(orig)
    plt.show()

