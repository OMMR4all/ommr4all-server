import os
import numpy as np
import torch
from PIL import Image
import torch
import torchvision
from torch import nn, Tensor

from typing import List, Tuple, Dict, Optional, Union

import omr.steps.layout.drop_capitals.transforms as T



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
class DropCapitalDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, additional_data=None, train = False):
        self.transforms = get_transform(True) if train else get_transform(False)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs
        self.masks = masks
        self.additional_data = additional_data

    def __getitem__(self, idx):
        # load images and masks
        img = self.imgs[idx]
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = self.masks[idx]
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        area = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if ymin -ymax == 0 or xmin -xmax == 0:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            area.append((xmax - xmin) * (ymax - ymin))
        area = torch.as_tensor(area, dtype=torch.float32)
        # convert everything into a torch.Tensor
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
