from typing import List, NamedTuple, Generator
from thirdparty.page_segmentation.lib.predictor import Predictor, PredictSettings
import os
from omr.datatypes import PcGts, Symbol, SymbolType, NoteComponent, Neume, Point
from omr.dataset.pcgtsdataset import PcGtsDataset, MusicLineAndMarkedSymbol
import cv2
import numpy as np

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
                ax[1].imshow(p.probabilities[:,:,0] > 0.8)
                ax[2].imshow(p.labels)
                ax[3].imshow((p.probabilities[:, :, 0] <= 0.8) * (1 + np.argmax(p.probabilities[:, :, 1:], axis=-1)))
                ax[4].imshow(p.data.mask)
                plt.show()
            m: MusicLineAndMarkedSymbol = p.data.user_data
            yield PredictionResult(self.exract_symbols(p.labels, m), p.data.user_data)

    def exract_symbols(self, p: np.ndarray, m: MusicLineAndMarkedSymbol) -> List[Symbol]:
        n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8))
        symbols = []
        for i in range(n_labels):
            coord = self.dataset.line_and_mask_operations.local_to_global_pos(
                Point(x=centroids[i, 0], y=centroids[i, 1]), m.operation.params)
            symbols.append(Neume(notes=[NoteComponent(coord=coord.astype(int))]))

        return symbols


if __name__ == '__main__':
    import main.book as book
    b = book.Book('Graduel')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[3:4]]
    pred = PCPredictor([b.local_path('pc_paths')])
    ps = list(pred.predict(val_pcgts))
    import matplotlib.pyplot as plt
    orig = np.array(ps[0].line.operation.page_image)
    for p in ps:
        for s in p.symbols:
            n: Neume = s
            for nc in n.notes:
                t, l = nc.coord.y, nc.coord.x
                orig[t - 2:t + 2, l-2:l+2] = 255

    plt.imshow(orig)
    plt.show()

