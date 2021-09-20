import os
from copy import deepcopy
from typing import Generator, List, Type, Dict

import numpy as np
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGenerator, SampleMeta, InputSample, \
    CalamariDataGeneratorParams, logger
from dataclasses import dataclass, field

from calamari_ocr.utils import glob_all
from calamari_ocr.utils.image import ImageLoader, ImageLoaderParams
from dataclasses_json.core import Json
from paiargparse import pai_dataclass, pai_meta
from paiargparse.dataclass_json_overrides import _decode_dataclass, _asdict
from tfaip import PipelineMode


def recover_to_dict(cls):
    print("2000" * 100)
    if hasattr(cls, '_to_dict'):
        print("heyehey")
        setattr(cls, 'to_dict', getattr(cls, '_to_dict'))
    return cls


@recover_to_dict
@pai_dataclass
@dataclass
class RawData(CalamariDataGeneratorParams):
    images: List[np.array] = field(default_factory=list, metadata=pai_meta(required=True))
    gt_files: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.images)

    def select(self, indices: List[int]):
        if self.images:
            self.images = [self.images[i] for i in indices]
        if self.gt_files:
            self.gt_files = [self.gt_files[i] for i in indices]

    def to_prediction(self):
        pred = deepcopy(self)
        return pred

    @staticmethod
    def cls():
        return RawDataReader

    def prepare_for_mode(self, mode: PipelineMode):
        self.images = self.images
        self.gt_files = self.gt_files

    def image_loader(self) -> ImageLoader:
        return ImageLoader(ImageLoaderParams())

    @classmethod
    def from_dict(cls, kvs, *, infer_missing=False):
        # Use custom _decode_dataclass with fixed types
        return _decode_dataclass(cls, kvs, infer_missing)

    def _to_dict(self, encode_json=False, include_cls=False) -> Dict[str, Json]:
        a = self.images
        self.images = []
        print("200" * 50)
        d = _asdict(self, encode_json=encode_json, include_cls=include_cls)
        self.images = a
        return d


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
