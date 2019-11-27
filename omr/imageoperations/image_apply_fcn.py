from . import ImageOperation, ImageOperationData, OperationOutput, ImageData
from copy import copy
from ocr4all_pixel_classifier.lib.predictor import Predictor, PredictSettings, Dataset, SingleData
import os
import numpy as np


class ImageApplyFCN(ImageOperation):
    def __init__(self, model: str):
        super().__init__()
        self.model = model
        # create the predictor if it is needed, it is possible that the model required is created later
        self.predictor = None

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        if self.predictor is None:
            settings = PredictSettings(
                network=os.path.splitext(self.model)[0]
            )
            self.predictor = Predictor(settings)

        d = data.images[0]
        data.images.append(ImageData(
            (list(self.predictor.predict(Dataset([SingleData(d.image)])))[0].probabilities * 255).astype(np.uint8),
            d.nearest_neighbour_rescale))
        return [data]

    def local_to_global_pos(self, p, params):
        return p
