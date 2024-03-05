import random

import cv2
import torch
import os
from omr.dataset import RegionLineMaskData
from omr.steps.layout.drop_capitals.dataset import DropCapitalDatasetDataset
from omr.steps.layout.predictor import LayoutAnalysisPredictor, PredictionType, PredictionResult, \
    PredictionCallback, AlgorithmPredictorSettings, FinalPredictionResult, IdCoordsPair
from typing import List, Optional
from database.file_formats.pcgts import PcGts, BlockType, Coords, Line, Rect, Point, Size, Page
import numpy as np
from omr.steps.layout.drop_capitals.meta import Meta
from scipy.spatial import ConvexHull, convex_hull_plot_2d

if __name__ == '__main__':
    import django
    import os

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

LAYOUT_DROP_CAPITAL_MODEL_DEFAULT_NAME = "layout_drop_capital.pt"
def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    # labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes


def draw_segmentation_map(image, masks, boxes):
    COLORS = np.random.uniform(0, 255, size=(30, 3))

    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                      thickness=2)
        # put the label text above the objects
        # cv2.putText(image, "DropCapital", (boxes[i][0][0], boxes[i][0][1] - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, color,
        #            thickness=2, lineType=cv2.LINE_AA)

    return image


def nothing(x):
    pass


def detect_red_areas(img):
    from matplotlib import pyplot as plt

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = img.copy()
    print(img_hsv[:, :, 2])
    print(img_hsv[:, :, 2].shape)
    img2[np.where(np.logical_and(img_hsv[:, :, 2] < 175, 140 < img_hsv[:, :, 2]))] = 0
    fig, ax = plt.subplots(1, 5, figsize=(15, 5), sharex=True, sharey=True)
    ax[0].imshow(img)
    ax[0].set_title('prig', fontsize=15)
    ax[1].imshow(img_hsv[:, :, 0], cmap='hsv')
    ax[1].set_title('Hue', fontsize=15)
    ax[2].imshow(img_hsv[:, :, 1], cmap='hsv')
    ax[2].set_title('Saturation', fontsize=15)
    ax[3].imshow(img_hsv[:, :, 2], cmap='hsv')
    ax[3].set_title('Value', fontsize=15)
    ax[4].imshow(img2)
    ax[4].set_title('prig', fontsize=15)
    plt.show()

    # lower mask (0-10)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([200, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    plt.imshow(output_hsv)
    plt.show()


def test(image):
    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while (1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')

        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


class DropCapitalPredictor(LayoutAnalysisPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torch.load(os.path.join(settings.model.local_file(LAYOUT_DROP_CAPITAL_MODEL_DEFAULT_NAME)),
                                map_location=torch.device(device))

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        dc_dataset = DropCapitalDatasetDataset(pcgts_files, self.dataset_params)
        dataset = dc_dataset.to_drop_capital_dataset()
        length = len(dataset.imgs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for x in range(length):
            image, target = dataset.__getitem__(x)
            add_data = dataset.additional_data[x]
            o_image = dataset.imgs[x]

            rlmd: RegionLineMaskData = add_data
            page: Page = add_data.operation.page
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

            yield PredictionResult(
                blocks={
                    BlockType.DROP_CAPITAL: coords,
                    BlockType.LYRICS: [],
                    BlockType.MUSIC: [],
                },
            )


if __name__ == '__main__':
    from database import DatabaseBook

    b = DatabaseBook('Pa_14819')

    # b = DatabaseBook('test3')
    # b = DatabaseBook('Cai_72')

    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[1:50]]

    pred = DropCapitalPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
