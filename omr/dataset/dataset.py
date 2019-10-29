from abc import ABC, abstractmethod
from types import MappingProxyType

from database.file_formats.pcgts import PcGts, Line, PageScaleReference, Point
import numpy as np
from typing import List, Tuple, Generator, Union, Optional, Any
from omr.dataset import RegionLineMaskData
from tqdm import tqdm
import logging

from omr.dataset.datastructs import CalamariCodec
from omr.imageoperations import ImageOperationList, ImageOperationData
from omr.dewarping.dummy_dewarper import NoStaffLinesAvailable, NoStaffsAvailable
from dataclasses import dataclass, field
from mashumaro import DataClassJSONMixin
from enum import Enum

import json


logger = logging.getLogger(__name__)


class DatasetCallback(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loading(self, n: int, total: int):
        pass

    @abstractmethod
    def loading_started(self, total: int):
        pass

    @abstractmethod
    def loading_finished(self, total: int):
        pass

    def apply(self, gen, total: int):
        self.loading_started(total)
        self.loading(0, total)
        for i, v in enumerate(gen):
            self.loading(i + 1, total)
            yield v
        self.loading_finished(total)


class ImageInput(Enum):
    LINE_IMAGE = 'line_image'
    REGION_IMAGE = 'region_image'


class LyricsNormalization(Enum):
    SYLLABLES = 'syllables'     # a-le-lu-ya vir-ga
    ONE_STRING = 'one_string'   # aleluyavirga
    WORDS = 'words'             # aleluya virga


@dataclass
class LyricsNormalizationParams(DataClassJSONMixin):
    lyrics_normalization: LyricsNormalization = LyricsNormalization.ONE_STRING
    lower_only: bool = True
    unified_u: bool = True
    remove_brackets: bool = True


class LyricsNormalizationProcessor:
    def __init__(self, params: LyricsNormalizationParams):
        self.params = params

    def apply(self, text: str) -> str:
        if self.params.lower_only:
            text = text.lower()

        if self.params.unified_u:
            text = text.replace('v', 'u')

        if self.params.remove_brackets:
            text = text.replace('<', '').replace('>', '')

        if self.params.lyrics_normalization == LyricsNormalization.ONE_STRING:
            text = text.replace('-', '').replace(' ', '')
        elif self.params.lyrics_normalization == LyricsNormalization.WORDS:
            text = text.replace('-', '')

        return text


@dataclass
class DatasetParams(DataClassJSONMixin):
    # general
    gt_required: bool = False
    pad: Optional[List[int]] = None
    pad_power_of_2: Optional[int] = 3
    image_input: ImageInput = ImageInput.LINE_IMAGE

    # staff line detection
    full_page: bool = True
    gray: bool = True
    extract_region_only: bool = True
    gt_line_thickness: int = 2
    page_scale_reference: PageScaleReference = PageScaleReference.NORMALIZED
    target_staff_line_distance: int = 10
    origin_staff_line_distance: int = 10

    # text
    lyrics_normalization: LyricsNormalizationParams = field(default_factory=lambda: LyricsNormalizationParams())

    # symbol detection
    height: int = 80
    dewarp: bool = False
    cut_region: bool = False
    center: bool = True
    staff_lines_only: bool = True
    masks_as_input: bool = False
    apply_fcn_pad_power_of_2: int = 3
    apply_fcn_model: Optional[str] = None
    apply_fcn_height: Optional[int] = None
    neume_types_only: bool = False
    calamari_codec: Optional[CalamariCodec] = None

    def mix_default(self, default_params: 'DatasetParams'):
        for key, value in default_params.to_dict().items():
            if getattr(self, key, None) is None:
                setattr(self, key, getattr(default_params, key))
            try:
                if getattr(self, key, -1) < 0:
                    setattr(self, key, getattr(default_params, key))
            except TypeError:
                pass


class Dataset(ABC):
    @staticmethod
    @abstractmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        pass

    def __init__(self, pcgts: List[PcGts],
                 params: DatasetParams,
                 ):
        self.params = params
        self.files = pcgts
        self.loaded: Optional[List[Tuple[Line, np.ndarray]]] = None
        self.image_ops = self.__class__.create_image_operation_list(self.params)

    def local_to_global_pos(self, p: Point, params: List[Any]) -> Point:
        return self.image_ops.local_to_global_pos(p, params)

    def to_page_segmentation_dataset(self, callback: Optional[DatasetCallback] = None):
        if self.params.origin_staff_line_distance == self.params.target_staff_line_distance:
            from pagesegmentation.lib.dataset import Dataset, SingleData
            return Dataset([SingleData(image=d.line_image if self.params.image_input == ImageInput.LINE_IMAGE else d.region,
                                       binary=((d.line_image < 125) * 255).astype(np.uint8),
                                       mask=d.mask,
                                       line_height_px=self.params.origin_staff_line_distance if self.params.origin_staff_line_distance is not None else d.operation.page.avg_staff_line_distance(),
                                       original_shape=d.line_image.shape,
                                       user_data=d) for d in self.load(callback)], {})
        else:
            raise NotImplementedError()

    def to_line_detection_dataset(self, callback: Optional[DatasetCallback] = None) -> List[RegionLineMaskData]:
        return self.load(callback)

    def to_calamari_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
        from calamari_ocr.ocr.datasets.dataset import RawDataSet, DataSetMode
        marked_symbols = self.load(callback)

        def get_input_image(d: RegionLineMaskData):
            if self.params.cut_region:
                return d.line_image.transpose()
            elif self.params.masks_as_input:
                hot = d.operation.images[3].image
                hot[:, :, 0] = (d.line_image if self.params.cut_region else d.region)
                return hot.transpose([1, 0, 2])
            else:
                return d.region.transpose()

        images = [get_input_image(d).astype(np.uint8) for d in marked_symbols]
        if self.params.neume_types_only:
            gts = [d.calamari_sequence(self.params.calamari_codec).calamari_neume_types_str for d in marked_symbols]
        else:
            gts = [d.calamari_sequence(self.params.calamari_codec).calamari_str for d in marked_symbols]
        return RawDataSet(DataSetMode.TRAIN if train else DataSetMode.PREDICT, images=images, texts=gts)

    def to_text_line_calamari_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
        from calamari_ocr.ocr.datasets.dataset import RawDataSet, DataSetMode
        lines = self.load(callback)

        def get_input_image(d: RegionLineMaskData):
            if self.params.cut_region:
                return d.line_image
            else:
                return d.region

        def extract_text(data: RegionLineMaskData):
            if not data.operation.text_line:
                return None

            text = data.operation.text_line.text(with_drop_capital=False)
            return LyricsNormalizationProcessor(self.params.lyrics_normalization).apply(text)

        images = [255 - get_input_image(d).astype(np.uint8) for d in lines]
        gts = [extract_text(d) for d in lines]
        return RawDataSet(DataSetMode.TRAIN if train else DataSetMode.PREDICT, images=images, texts=gts)

    def load(self, callback: Optional[DatasetCallback] = None) -> List[RegionLineMaskData]:
        if self.loaded is None:
            self.loaded = list(self._load(callback))

        return self.loaded

    def _load(self, callback: Optional[DatasetCallback]) -> Generator[RegionLineMaskData, None, None]:
        def wrapper(g):
            if callback:
                return callback.apply(g, total=len(self.files))
            return g

        for f in tqdm(wrapper(self.files), total=len(self.files), desc="Loading Dataset"):
            try:
                input = ImageOperationData([], self.params.page_scale_reference, page=f.page, pcgts=f)
                for outputs in self.image_ops.apply_single(input):
                    yield RegionLineMaskData(outputs)
            except (NoStaffsAvailable, NoStaffLinesAvailable):
                pass
            except Exception as e:
                logger.exception("Exception during processing of page: {}".format(f.page.location.local_path()))
                raise e
