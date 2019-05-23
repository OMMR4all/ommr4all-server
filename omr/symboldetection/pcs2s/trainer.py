from typing import Optional
from omr.symboldetection.dataset import SymbolDetectionDataset, SymbolDetectionDatasetParams
from omr.symboldetection.trainer import SymbolDetectionTrainerCallback, SymbolDetectionTrainerBase, SymbolDetectionTrainerParams, CalamariParams
from omr.symboldetection.pixelclassifier.trainer import PCTrainer
from omr.symboldetection.sequencetosequence.trainer import OMRTrainer
from database import DatabaseBook
import os


class PCS2STrainer(SymbolDetectionTrainerBase):
    def __init__(self, params: SymbolDetectionTrainerParams):
        ds_params = params.dataset_params
        ds_params.masks_as_input = True
        ds_params.apply_fcn_height = ds_params.height
        ds_params.apply_fcn_model = os.path.join(params.output, 'pc_model')
        params.calamari_params.channels = 9

        super().__init__(params)

        pc_ds_params = SymbolDetectionDatasetParams(
            gt_required=True,
            height=ds_params.apply_fcn_height,
            dewarp=ds_params.dewarp,
            cut_region=ds_params.cut_region,
            pad=ds_params.pad,
            pad_power_of_2=ds_params.apply_fcn_pad_power_of_2,
            center=ds_params.center,
            staff_lines_only=ds_params.staff_lines_only,
            masks_as_input=False,
        )
        pc_train_params = SymbolDetectionTrainerParams(
            dataset_params=pc_ds_params,
            train_data=params.train_data,
            validation_data=params.validation_data,
            output=ds_params.apply_fcn_model,
            page_segmentation_params=params.page_segmentation_params,
        )

        self.pc_trainer = PCTrainer(pc_train_params)
        self.s2s_trainer = OMRTrainer(self.params)

    def run(self, model_for_book: Optional[DatabaseBook] = None, callback: Optional[SymbolDetectionTrainerCallback] = None):
        print("Training the pixel classifier")
        #self.pc_trainer.run(model_for_book, callback)
        print("Training Calamari")
        self.s2s_trainer.run(model_for_book, callback)
        print("Done")


if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    b = DatabaseBook('Graduel_Fully_Annotated')
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState("Symbols", True), LockState("Layout", True)], True, [b])
    output = 'models_out/test_pcs2s'
    params = SymbolDetectionDatasetParams(
        gt_required=True,
        height=40,
        dewarp=True,
        cut_region=False,
        pad=(0, 10, 0, 20),
        pad_power_of_2=None,
        center=True,
        staff_lines_only=True,
    )
    train_params = SymbolDetectionTrainerParams(
        params,
        train_pcgts,
        val_pcgts,
        output=output,
        l_rate=1e-3,
        early_stopping_test_interval=1000,
        calamari_params=CalamariParams(
            network='cnn=40:3x3,pool=1x2,cnn=80:3x3,lstm=100,dropout=0.5',
            n_folds=0,
        )
    )
    trainer = PCS2STrainer(train_params)
    trainer.run(b)



