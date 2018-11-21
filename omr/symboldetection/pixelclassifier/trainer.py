from typing import List
from omr.datatypes import PcGts
import main.book as book
from omr.dataset.pcgtsdataset import PcGtsDataset
import os
from omr.imageoperations.music_line_operations import SymbolLabel

class PCTrainer:
    def __init__(self, train_pcgts_files: List[PcGts], validation_pcgts_files: List[PcGts]):
        self.height = 80
        self.train_pcgts_dataset = PcGtsDataset(train_pcgts_files, gt_required=True, height=self.height)
        self.validation_pcgts_dataset = PcGtsDataset(validation_pcgts_files, gt_required=True, height=self.height)

    def run(self, model_for_book: book.Book):
        from thirdparty.page_segmentation.pagesegmentation.lib.trainer import Trainer, TrainSettings
        a = self.train_pcgts_dataset.to_music_line_page_segmentation_dataset()

        settings = TrainSettings(
            n_iter=20000,
            n_classes=len(SymbolLabel),
            l_rate=1e-4,
            train_data=self.train_pcgts_dataset.to_music_line_page_segmentation_dataset(),
            validation_data=self.validation_pcgts_dataset.to_music_line_page_segmentation_dataset(),
            load=None,
            display=100,
            output=model_for_book.local_path(os.path.join('pc_paths', 'model')),
            early_stopping_test_interval=1000,
            early_stopping_max_keep=5,
            early_stopping_on_accuracy=True,
            threads=4,
        )

        trainer = Trainer(settings)
        trainer.train()

        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    b = book.Book('Graduel')
    pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[1:4]]
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:1]]
    page = book.Book('Graduel').page('Graduel_de_leglise_de_Nevers_023')
    # pcgts = PcGts.from_file(page.file('pcgts'))
    trainer = PCTrainer(pcgts, val_pcgts)
    trainer.run(b)

