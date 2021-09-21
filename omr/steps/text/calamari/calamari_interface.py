import os
from typing import Generator, List, Type, Dict, Mapping, Collection

import dataclasses_json
import numpy as np
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGenerator, SampleMeta, InputSample, \
    CalamariDataGeneratorParams, logger
from dataclasses import dataclass, field, is_dataclass, _is_dataclass_instance, fields

from calamari_ocr.utils import glob_all
from calamari_ocr.utils.image import ImageLoader, ImageLoaderParams
from paiargparse import pai_dataclass, pai_meta

from tfaip import PipelineMode
import copy
from dataclasses_json import dataclass_json, config


@pai_dataclass
@dataclass
class RawData(CalamariDataGeneratorParams):
    images: List[np.array] = field(default_factory=list, metadata=config(encoder=lambda x: None))
    gt_files: List[str] = field(default_factory=list, metadata=config(encoder=lambda x: None))

    def __len__(self):
        return len(self.images)

    def select(self, indices: List[int]):
        if self.images:
            self.images = [self.images[i] for i in indices]
        if self.gt_files:
            self.gt_files = [self.gt_files[i] for i in indices]

    def to_prediction(self):
        pred = copy.deepcopy(self)
        return pred

    @staticmethod
    def cls():
        return RawDataReader

    def prepare_for_mode(self, mode: PipelineMode):
        self.images = self.images
        self.gt_files = self.gt_files

    def image_loader(self) -> ImageLoader:
        return ImageLoader(ImageLoaderParams())


class RawDataReader(CalamariDataGenerator):
    def __init__(self, mode: PipelineMode, params: RawData):
        """Create a dataset from memory

        Since this dataset already contains all data in the memory, this dataset may not be loaded

        Parameters
        ----------
        images : list of images
            the images of the dataset
        texts : list of str
            the texts of this dataset
        """

        super().__init__(mode, params=params)
        if mode == PipelineMode.PREDICTION:
            texts = [None] * len(params.images)
        else:
            texts = params.gt_files

        if mode == PipelineMode.TARGETS:
            images = [None] * len(params.texts)
        else:
            images = params.images

        for id, (image, text) in enumerate(zip(images, texts)):
            try:
                if image is None and text is None:
                    raise Exception("An empty data point is not allowed. Both image and text file are None")

            except Exception as e:
                logger.exception(e)
                if self.params.skip_invalid:
                    logger.warning("Exception raised. Invalid data. Skipping invalid example.")
                    continue
                else:
                    raise e

            self.add_sample(
                {
                    "image": image,
                    "text": text,
                    "id": id,
                }
            )

        self.loaded = True

    # def populate_folds(self, n_folds):
    #    super(RawDataReader, self).populate_folds(n_folds)
    #    for s in self.samples():
    #        s["meta"].fold_id = s["fold_id"]

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        if text_only:
            yield InputSample(None, sample["text"], SampleMeta(sample["id"], fold_id=sample["fold_id"]))
        yield InputSample(sample["image"], sample["text"], SampleMeta(sample["id"], fold_id=sample["fold_id"]))
