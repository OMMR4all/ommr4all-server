import os
from pathlib import Path

import albumentations
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from segmentation.datasets.dataset import MemoryDataset
from torch.utils.data import DataLoader

from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.stafflines.detection.trainer import StaffLineDetectionTrainer
from omr.steps.symboldetection.torchpixelclassifier.params import remove_nones
from segmentation.callback import ModelWriterCallback
# from segmentation.dataset import MemoryDataset
from segmentation.losses import Losses
from segmentation.metrics import Metrics
from segmentation.model_builder import ModelBuilderMeta
from segmentation.network import NetworkTrainer
from segmentation.optimizer import Optimizers
from segmentation.preprocessing.workflow import GrayToRGBTransform, ColorMapTransform, NetworkEncoderTransform, \
    PreprocessingTransforms
from segmentation.scripts.train import get_default_device
from segmentation.settings import ColorMap, ClassSpec, Preprocessingfunction, PredefinedNetworkSettings, \
    CustomModelSettings, ModelConfiguration, ProcessingSettings, NetworkTrainSettings

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from database.file_formats.performance.pageprogress import Locks, LockState
from database import DatabaseBook
import logging
from omr.steps.stafflines.detection.dataset import DatasetParams
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.steps.stafflines.detection.pixelclassifier_torch.meta import Meta
from typing import Optional, List

logger = logging.getLogger(__name__)


class BasicStaffLinesTrainerTorch(StaffLineDetectionTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=2000,
            l_rate=1e-3,
            display=10,
            early_stopping_max_keep=10,
            early_stopping_test_interval=200,
        )

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        if callback:
            callback.resolving_files()
        print(self.settings)
        train_data = self.train_dataset.to_memory_dataset(callback)

        color_map = ColorMap([ClassSpec(label=0, name="background", color=[255, 255, 255]),
                              ClassSpec(label=1, name="line", color=[255, 0, 0])])

        input_transforms = albumentations.Compose(remove_nones([
            GrayToRGBTransform() if True else None,
            ColorMapTransform(color_map=color_map.to_albumentation_color_map())

        ]))
        aug_transforms = self.settings.page_segmentation_torch_params.augmentation \
            if self.settings.page_segmentation_torch_params.data_augmentation else None
        tta_transforms = None
        post_transforms = albumentations.Compose(remove_nones([
            NetworkEncoderTransform(
                self.settings.page_segmentation_torch_params.encoder if not self.settings.page_segmentation_torch_params.custom_model else Preprocessingfunction.name),
            ToTensorV2()
        ]))
        transforms = PreprocessingTransforms(
            input_transform=input_transforms,
            aug_transform=aug_transforms,
            # tta_transforms=tta_transforms,
            post_transforms=post_transforms,
        )

        train_data = MemoryDataset(df=train_data, transforms=transforms.get_train_transforms(), scale_area=9999999999)
        val_data = MemoryDataset(self.validation_dataset.to_memory_dataset(callback),
                                 transforms=transforms.get_test_transforms(), scale_area=9999999999)
        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

        predfined_nw_settings = PredefinedNetworkSettings(
            architecture=self.settings.page_segmentation_torch_params.architecture,
            encoder=self.settings.page_segmentation_torch_params.encoder,
            classes=len(color_map),
            encoder_depth=self.settings.page_segmentation_torch_params.predefined_encoder_depth,
            decoder_channel=self.settings.page_segmentation_torch_params.predefined_decoder_channel)
        custom_nw_settings = CustomModelSettings(
            encoder_filter=self.settings.page_segmentation_torch_params.custom_model_encoder_filter,
            decoder_filter=self.settings.page_segmentation_torch_params.custom_model_encoder_filter,
            attention_encoder_filter=[12, 32, 64, 128],
            attention=CustomModelSettings.attention,
            classes=len(color_map),
            attention_depth=CustomModelSettings.attention_depth,
            encoder_depth=self.settings.page_segmentation_torch_params.custom_model_encoder_depth,
            attention_encoder_depth=CustomModelSettings.attention_encoder_depth,
            stride=CustomModelSettings.stride,
            padding=CustomModelSettings.padding,
            kernel_size=CustomModelSettings.kernel_size,
            weight_sharing=False if CustomModelSettings.weight_sharing else True,
            scaled_image_input=CustomModelSettings.scaled_image_input
        )

        config = ModelConfiguration(use_custom_model=self.settings.page_segmentation_torch_params.custom_model,
                                    network_settings=predfined_nw_settings if not self.settings.page_segmentation_torch_params.custom_model else None,
                                    custom_model_settings=custom_nw_settings if self.settings.page_segmentation_torch_params.custom_model else None,
                                    preprocessing_settings=ProcessingSettings(input_padding_value=32,
                                                                              rgb=True,
                                                                              scale_max_area=999999999,
                                                                              preprocessing=Preprocessingfunction(
                                                                                  self.settings.page_segmentation_torch_params.encoder) if not self.settings.page_segmentation_torch_params.custom_model else Preprocessingfunction(),
                                                                              transforms=transforms.to_dict()),

                                    color_map=color_map)
        network = ModelBuilderMeta(config, device=get_default_device()).get_model()
        mw = ModelWriterCallback(network, config, save_path=Path(self.settings.model.path), prefix="",
                                 metric_watcher_index=0)
        callbacks = [mw]
        trainer = NetworkTrainer(network, NetworkTrainSettings(classes=len(color_map),
                                                               optimizer=Optimizers("adam"),
                                                               learningrate_seghead=self.params.l_rate,
                                                               learningrate_encoder=self.params.l_rate,
                                                               learningrate_decoder=self.params.l_rate,
                                                               batch_accumulation=1,
                                                               processes=self.params.processes,
                                                               metrics=[Metrics("accuracy")],
                                                               watcher_metric_index=0,
                                                               loss=Losses.cross_entropy_loss,
                                                               ), get_default_device(),
                                 callbacks=callbacks, debug_color_map=config.color_map)

        os.makedirs(os.path.dirname(self.settings.model.path), exist_ok=True)
        trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, n_epoch=50, lr_schedule=None)


if __name__ == "__main__":
    from database import DatabaseBook
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    b = DatabaseBook('Graduel_Part_1_gt')
    c = DatabaseBook('Graduel_Part_2_gt')
    d = DatabaseBook('Graduel_Part_3_gt')
    e = DatabaseBook('Pa_14819_gt')
    f = DatabaseBook('Assisi')
    g = DatabaseBook('Cai_72')
    train, val = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True)], datasets=[b, c, d, e, f, g])
    trainer = BasicStaffLinesTrainerTorch(AlgorithmTrainerSettings(
        dataset_params=DatasetParams(),
        train_data=train,
        validation_data=val,
    ))
    trainer.train(b)
