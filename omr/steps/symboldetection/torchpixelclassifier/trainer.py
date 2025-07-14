import os
from pathlib import Path

import albumentations
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from segmentation.datasets.dataset import MemoryDataset

from omr.steps.callback import SegmentationProcessTrainCallback
from omr.steps.symboldetection.torchpixelclassifier.params import remove_nones
from segmentation.callback import ModelWriterCallback
from segmentation.losses import Losses
from segmentation.metrics import Metrics, MetricReduction
from segmentation.model_builder import ModelBuilderMeta, ModelBuilderLoad
from segmentation.optimizer import Optimizers
from segmentation.preprocessing.workflow import GrayToRGBTransform, ColorMapTransform, PreprocessingTransforms, \
    NetworkEncoderTransform
from segmentation.scripts.train import get_default_device
from segmentation.settings import ModelConfiguration, ClassSpec, ColorMap, PredefinedNetworkSettings, \
    CustomModelSettings, ProcessingSettings, Preprocessingfunction, NetworkTrainSettings
from torch.utils.data import DataLoader

from omr.dataset import DatasetParams
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional, Tuple
from database import DatabaseBook

from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.imageoperations.music_line_operations import SymbolLabel
# from ocr4all_pixel_classifier.lib.trainer import Trainer, Loss, Monitor, Architecture
from omr.steps.symboldetection.torchpixelclassifier.meta import Meta
from segmentation.network import Network, NetworkTrainer
# from segmentation.dataset import MemoryDataset

from torch.nn.utils.rnn import pad_sequence


def build_model_from_loaded(load, device) -> Tuple[Network, ModelConfiguration]:
    base_model_file = ModelBuilderLoad.from_disk(model_weights=load, device=device)
    base_model = base_model_file.get_model()
    base_config = base_model_file.get_model_configuration()
    return base_model, base_config


class PCTorchTrainer(SymbolDetectionTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=20000,
            l_rate=1e-4,
            display=100,
            early_stopping_test_interval=500,
            early_stopping_max_keep=10,
            processes=1,
        )

    @staticmethod
    def default_dataset_params() -> DatasetParams:
        return DatasetParams(
            pad=[0, 10, 0, 40],
            dewarp=False,
            center=False,
            staff_lines_only=True,
            cut_region=False,
            keep_graphical_connection=[True, False, True] #Todo
        )

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        params.pad_power_of_2 = False
        params.center = False

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        #pc_callback = PCTorchTrainerCallback(callback) if callback else None
        pc_callback = SegmentationProcessTrainCallback(callback, epochs=self.params.n_epoch) if callback else None
        if callback:
            callback.resolving_files()

        train_data_pd = self.train_dataset.to_memory_dataset(callback, same_dim=False)
        from matplotlib import pyplot as plt

        color_map = ColorMap([ClassSpec(label=i.value, name=i.name.lower(), color=i.get_color()) for i in SymbolLabel])
        input_transforms = Compose(remove_nones([
            GrayToRGBTransform() if True else None,
            ColorMapTransform(color_map=color_map.to_albumentation_color_map())

        ]))
        aug_transforms = self.settings.page_segmentation_torch_params.augmentation \
            if self.settings.page_segmentation_torch_params.data_augmentation else None
        tta_transforms = None
        post_transforms = Compose(remove_nones([
            # NetworkEncoderTransform(Preprocessingfunction.name),
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
        val_data_pd = self.validation_dataset.to_memory_dataset(callback)
        train_data = MemoryDataset(df=train_data_pd, transforms=transforms.get_train_transforms())
        val_data = MemoryDataset(df=val_data_pd, transforms=transforms.get_test_transforms())

        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

        predfined_nw_settings = PredefinedNetworkSettings(
            architecture=self.settings.page_segmentation_torch_params.architecture,
            encoder=self.settings.page_segmentation_torch_params.encoder,
            classes=len(SymbolLabel),
            encoder_depth=self.settings.page_segmentation_torch_params.predefined_encoder_depth,
            decoder_channel=self.settings.page_segmentation_torch_params.predefined_decoder_channel,
            use_batch_norm_layer=self.settings.page_segmentation_torch_params.use_batch_norm_layer)
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
        if self.params.model_to_load():
            path = self.params.model_to_load().local_file('best.torch')
            device = get_default_device()
            network, config = build_model_from_loaded(path, device)
        else:
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
        callbacks = [mw, pc_callback]
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
        trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, n_epoch=self.params.n_epoch, lr_schedule=None)


if __name__ == '__main__':
    from omr.dataset import DatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    # b = DatabaseBook('Graduel_Part_1_gt')
    # c = DatabaseBook('Graduel_Part_2_gt')
    # d = DatabaseBook('Graduel_Part_3_gt')
    # e = DatabaseBook('Pa_14819_gt')
    # f = DatabaseBook('Assisi')
    # g = DatabaseBook('Cai_72')
    # h = DatabaseBook('pa_904')
    #i = DatabaseBook('mul_2_rsync_gt2')
    j = DatabaseBook('Geesebook1gt')
    k = DatabaseBook('Geesebook2gt')
    l = DatabaseBook('Winterburger')

    train, val = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True)], datasets=[j, k, l])
    settings = AlgorithmTrainerSettings(
        DatasetParams(),
        train ,
        val,
        params=AlgorithmTrainerParams(load="e/Graduel_Part_1_gt/symbols_pc_torch/2023-02-02T14:01:14")
    )
    trainer = PCTorchTrainer(settings)
    trainer.train(j)
