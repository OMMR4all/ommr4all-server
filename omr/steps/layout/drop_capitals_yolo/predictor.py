import random
import os
import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from omr.dataset import RegionLineMaskData
from omr.steps.layout.drop_capitals_yolo.dataset import DropCapitalDatasetDataset
from omr.steps.layout.predictor import LayoutAnalysisPredictor, PredictionType, PredictionResult, \
    PredictionCallback, AlgorithmPredictorSettings, FinalPredictionResult, IdCoordsPair
from typing import List, Optional
from database.file_formats.pcgts import PcGts, BlockType, Coords, Line, Rect, Point, Size, Page, PageScaleReference
import numpy as np
from omr.steps.layout.drop_capitals_yolo.meta import Meta
from loguru import logger

if __name__ == '__main__':
    import django
    import os

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()




    cv2.destroyAllWindows()

def rescale_image_bigger_side_to(image, size):
    h, w = image.shape[:2]
    if h > w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    return cv2.resize(image, (new_w, new_h))
class DropCapitalPredictor(LayoutAnalysisPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.model = torch.load(os.path.join(settings.model.local_file(LAYOUT_DROP_CAPITAL_MODEL_DEFAULT_NAME)),
        #                        map_location=torch.device(device))
        #self.model = YOLO("/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/runs/detect/train/weights/best.pt")
        #self.model = YOLO("/home/alexanderh/Downloads/yolov8n_layout_camerarius.pt")
        #self.model = YOLO("/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/runs/detect/train16/weights/best.pt")
        path = os.path.join(settings.model.path)
        logger.info(f'Using model: {path}')

        model_weights = os.path.join(path, 'best.pt')
        self.model = YOLO(model_weights)
    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        dc_dataset = DropCapitalDatasetDataset(pcgts_files, self.dataset_params)
        images, masks, adds = dc_dataset.to_yolo_drop_capital_dataset(train_path="")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        index = 0



        for image, mask, add in zip(images, masks, adds):
            print(pcgts_files[index].page.location.page)
            index += 1
            image2 = image[:, :, [2, 1, 0]]

            #image = rescale_image_bigger_side_to(image, 960)
            #print(image.shape)
            rlmd: RegionLineMaskData = add
            page: Page = add.operation.page
            output = self.model.predict(image2)

            def transform_points(box):
                xyxy = box.cpu().xyxy[0]
                x1 = xyxy[0]
                x2 = xyxy[2]
                y1 = xyxy[1]
                y2 = xyxy[3]
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                coords_tr = Coords(
                    np.array([dc_dataset.local_to_global_pos(Point(p[0], p[1]), rlmd.operation.params).p for p in
                              points]))
                return coords_tr
            coords = []

            for r in output:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    coords.append(page.image_to_page_scale(transform_points(box),
                                                           rlmd.operation.scale_reference))
                if False:
                    annotator = Annotator(image)
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]
                        c = box.cls
                        annotator.box_label(b, "")
                    img = annotator.result()
                    from matplotlib import pyplot as plt
                    plt.imshow(img)
                    plt.show()
            """
            image = image.unsqueeze(0).to(device)
            
            masks, boxes = get_outputs(image, self.model, 0.5)
            coords = []
            for mask, box in zip(masks, boxes):
                points = np.argwhere(mask == True)
                if len(points) == 0:
                    continue
                hull = ConvexHull(points)
                convex_hull_points = hull.points[hull.vertices]

                def transform_points(yx_points):
                    return Coords(np.array(
                        [dc_dataset.local_to_global_pos(Point(p[1], p[0]), rlmd.operation.params).p for p in
                         yx_points]))

                coords.append(page.image_to_page_scale(transform_points(list(convex_hull_points)),
                                                       rlmd.operation.scale_reference))
            """
            yield PredictionResult(
                blocks={
                    BlockType.DROP_CAPITAL: coords,
                    BlockType.LYRICS: [],
                    BlockType.MUSIC: [],
                },
            )


if __name__ == '__main__':
    from database import DatabaseBook

    b = DatabaseBook('Aveiro_ANTF28')

    # b = DatabaseBook('test3')
    # b = DatabaseBook('Cai_72')

    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()]
    print(val_pcgts)
    pred = DropCapitalPredictor(AlgorithmPredictorSettings(Meta.default_model_for_book(b)))
    locations = [p.page.location for p in val_pcgts]
    #ps = list(pred.predict([p.page.location for p in val_pcgts]))
    from PIL import Image
    for index, i in enumerate(pred.predict(locations)):
        loc = locations[index]
        coords = i.blocks[BlockType.DROP_CAPITAL]
        image = np.array(Image.open(loc.pcgts().dataset_page().file('color_norm_x2').local_path()))
        for t in coords:
            print(t)
            loc.pcgts().page.page_to_image_scale(t.coords, PageScaleReference.NORMALIZED_X2).draw(image)
        from matplotlib import pyplot as plt
        plt.imshow(image)
        plt.show()
        print(i)

