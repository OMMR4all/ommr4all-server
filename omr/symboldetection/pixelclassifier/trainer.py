from typing import List
from omr.datatypes import PcGts
import main.book as book
from omr.dataset.pcgtsdataset import PcGtsDataset
import os
from omr.imageoperations.music_line_operations import SymbolLabel
from pagesegmentation.lib.trainer import Trainer, TrainSettings, TrainProgressCallback
from omr.symboldetection.trainer import SymbolDetectionTrainerCallback
import omr.symboldetection.pixelclassifier.settings as pc_settings

class PCTrainerCallback(TrainProgressCallback):
    def __init__(self, callback: SymbolDetectionTrainerCallback):
        super().__init__()
        self.callback = callback

    def init(self, total_iters, early_stopping_iters):
        self.callback.init(total_iters, early_stopping_iters)

    def next_iteration(self, iter: int, loss: float, acc: float, fgpa: float):
        self.callback.next_iteration(iter, loss, acc)

    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        self.callback.next_best_model(best_iter, best_acc, best_iters)

    def early_stopping(self):
        self.callback.early_stopping()


class PCTrainer:
    def __init__(self, train_pcgts_files: List[PcGts], validation_pcgts_files: List[PcGts]):
        self.height = 80
        self.train_pcgts_dataset = PcGtsDataset(train_pcgts_files, gt_required=True, height=self.height)
        self.validation_pcgts_dataset = PcGtsDataset(validation_pcgts_files, gt_required=True, height=self.height)

    def run(self, model_for_book: book.Book, callback: SymbolDetectionTrainerCallback = None):
        pc_callback = PCTrainerCallback(callback) if callback else None

        settings = TrainSettings(
            n_iter=20000,
            n_classes=len(SymbolLabel),
            l_rate=1e-4,
            train_data=self.train_pcgts_dataset.to_music_line_page_segmentation_dataset(),
            validation_data=self.validation_pcgts_dataset.to_music_line_page_segmentation_dataset(),
            load=None,
            display=100,
            output=model_for_book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name)),
            early_stopping_test_interval=500,
            early_stopping_max_keep=5,
            early_stopping_on_accuracy=True,
            threads=4,
        )

        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    b = book.Book('Graduel')
    pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[12:13]]
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:1]]
    page = book.Book('Graduel').page('Graduel_de_leglise_de_Nevers_033')
    # pcgts = PcGts.from_file(page.file('pcgts'))
    trainer = PCTrainer(pcgts, val_pcgts)
    trainer.run(b)

