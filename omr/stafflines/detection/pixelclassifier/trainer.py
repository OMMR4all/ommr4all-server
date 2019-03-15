from omr.stafflines.detection.trainer import StaffLinesTrainer
from omr.dataset.datafiles import dataset_by_locked_pages, EmptyDataSetException
from database import DatabaseBook
from database.file_formats import PcGts
import numpy as np
import os
import logging
from typing import List
from pagesegmentation.lib.dataset import SingleData, Dataset
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
from PIL import Image
import omr.stafflines.detection.pixelclassifier.settings as pc_settings
from omr.imageoperations import ImagePadToPowerOf2, ImageScaleOperation, ImageOperationData, ImageData, ImageOperationList

logger = logging.getLogger(__name__)


def pcgts_files_to_dataset(files: List[PcGts]) -> Dataset:
    l = []
    for f in files:
        logging.debug("Loading {}".format(f.page.location.local_path()))
        img_ops = ImageOperationList([
            ImageScaleOperation(0.5),
            ImagePadToPowerOf2(),
        ])
        gray_path = f.page.location.file('gray_deskewed', create_if_not_existing=True).local_path()
        binary_path = f.page.location.file('binary_deskewed', create_if_not_existing=True).local_path()
        image = np.array(Image.open(gray_path))
        binary = np.array(Image.open(binary_path))

        mask = np.zeros(image.shape, dtype=np.uint8)
        line_thickness = 5
        for mr in f.page.music_regions:
            for staff in mr.staffs:
                for staff_line in staff.staff_lines:
                    staff_line.draw(mask, (1, ), line_thickness)

        image, binary, mask = tuple(map(lambda x: x.image, img_ops.apply_single(ImageOperationData([
            ImageData(image, False), ImageData(binary, True), ImageData(mask, False)]))[0]))

        image = (255 - image).astype(np.uint8)
        binary = (1 - binary / 255).astype(np.uint8)
        mask = mask.astype(np.uint8)

        if False:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(1, 3)
            ax[0].imshow(image)
            ax[1].imshow(mask)
            plt.show()

        assert(mask.max() <= 1)

        l.append(SingleData(
            image=image,
            binary=binary,
            mask=mask,
            line_height_px=10,
            original_shape=image.shape,
            xpad=0,
            ypad=0,
            user_data=None,
        ))

    return Dataset(l)


class BasicStaffLinesTrainer(StaffLinesTrainer):
    def __init__(self, b: DatabaseBook):
        super().__init__(b)

    def train(self):
        train, val = dataset_by_locked_pages(0.8, 'CreateStaffLines')
        if len(train) == 0 or len(val) == 0:
            raise EmptyDataSetException()

        from pagesegmentation.lib.trainer import TrainSettings, Trainer

        settings = TrainSettings(
            n_iter=1000,
            n_classes=2,
            l_rate=1e-3,
            train_data=pcgts_files_to_dataset(train),
            validation_data=pcgts_files_to_dataset(val),
            load=None,
            display=100,
            output=self.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name)),
            early_stopping_test_interval=50,
            early_stopping_max_keep=10,
            early_stopping_on_accuracy=True,
            threads=8,
            data_augmentation=DefaultAugmenter(angle=(-2, 2), flip=(0.5, 0.5), contrast=0.8, brightness=20),
        )
        trainer = Trainer(settings)
        trainer.train()


if __name__=="__main__":
    trainer = BasicStaffLinesTrainer(DatabaseBook('Graduel'))
    trainer.train()
