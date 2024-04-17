import os
import tempfile
from pathlib import Path

import torch
import yaml
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from omr.dataset import DatasetParams
from omr.steps.layout.drop_capitals.torch_dataset import DropCapitalDataset
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional, Type, Dict
from database import DatabaseBook

from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import AlgorithmTrainer, TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from omr.steps.layout.drop_capitals_yolo.meta import Meta

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
class DropCapitalTrainer(SymbolDetectionTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=10000,
            l_rate=1e-4,
            display=100,
            early_stopping_test_interval=500,
            early_stopping_max_keep=10,
            processes=1,
        )

    @staticmethod
    def default_dataset_params() -> DatasetParams:
        return DatasetParams(
        )

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        params.pad_power_of_2 = True

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        if callback:
            callback.resolving_files()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if callback:
            callback.resolving_files()

        def make_yaml(train: Path, val: Path, yaml_fname) -> Dict[int, str]:
            root = train.parent
            lookup_dict = {0: "Drop Capital"}

            yaml_dict = {
                "path": str(root.absolute()),
                "train": str(train.name),
                "val": str(val.name),
                "names": lookup_dict,
            }
            with open(yaml_fname, 'w') as f:
                yaml.dump(yaml_dict, f)
            return lookup_dict

        with tempfile.TemporaryDirectory() as dirpath:
            print(dirpath)
            os.mkdir(os.path.join(dirpath, "train"))
            os.mkdir(os.path.join(dirpath, "val"))
            train_path = Path(os.path.join(dirpath, "train"))
            val_path = Path(os.path.join(dirpath, "val"))
            yaml_path = Path(os.path.join(dirpath, "data.yaml"))
            look_up = make_yaml(train_path, val_path, yaml_path)
            train_dataset = self.train_dataset.to_yolo_drop_capital_dataset(train=True, train_path=train_path,
                                                                      callback=callback)
            val_dataset = self.validation_dataset.to_yolo_drop_capital_dataset(train=True, train_path=val_path,
                                                                         callback=callback)

            from ultralytics import YOLO

            # Load the model.
            model = YOLO("/home/alexanderh/Downloads/yolov8n_layout_camerarius.pt")

            # Training.
            results = model.train(
                data=yaml_path,
                #name='yolov8m.pt',
                save=True,
                epochs=1000,
                imgsz=960,
                fliplr=0.5,
                scale=0.1,
                degrees=2,

            )
        "/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/runs/detect/train/weights/best.pt"
        # val_dataset = self.validation_dataset.to_nautilus_dataset(train=False, callback=callback)




if __name__ == '__main__':
    from omr.dataset import DatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    b = DatabaseBook('Pa_14819')
    c = DatabaseBook('Aveiro_ANTF28')

    train, val = dataset_by_locked_pages(0.8, [LockState(Locks.LAYOUT, True)], datasets=[b, c])
    print(len(train))
    print(len(val))
    settings = AlgorithmTrainerSettings(
        DatasetParams(),
        train,
        val,

    )
    trainer = DropCapitalTrainer(settings)
    trainer.train(b)

