import os
import string
from abc import ABC, abstractmethod
from types import MappingProxyType

import pandas as pd

#from tfaip import PipelineMode

from database.file_formats.pcgts import PcGts, Line, PageScaleReference, Point, BlockType
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
    SYLLABLES = 'syllables'  # a-le-lu-ya vir-ga
    ONE_STRING = 'one_string'  # aleluyavirga
    WORDS = 'words'  # aleluya virga


@dataclass
class LyricsNormalizationParams(DataClassJSONMixin):
    lyrics_normalization: LyricsNormalization = LyricsNormalization.WORDS
    lower_only: bool = True
    unified_u: bool = False
    remove_brackets: bool = True
    remove_punctuation: bool = False


class LyricsNormalizationProcessor:
    def __init__(self, params: LyricsNormalizationParams):
        self.params = params

    def apply(self, text: str) -> str:
        if self.params.lower_only:
            text = text.lower()
        if self.params.remove_punctuation:
            for i in string.punctuation:
                text = text.replace(i, '')
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
    text_types: List[BlockType] = field(default_factory=lambda:  [BlockType.LYRICS, BlockType.DROP_CAPITAL, BlockType.PARAGRAPH])#= #["lyrics", "dropCapital"]

    # symbol detection
    height: int = 80
    dewarp: bool = False
    cut_region: bool = False
    center: bool = True
    staff_lines_only: bool = True
    keep_graphical_connection: Optional[List[bool]] = None # [ns, gapped, looped] all note graphical connection are merged to gapped type if true
    masks_as_input: bool = False
    apply_fcn_pad_power_of_2: int = 3
    apply_fcn_model: Optional[str] = None
    apply_fcn_height: Optional[int] = None
    neume_types_only: bool = False
    calamari_codec: Optional[CalamariCodec] = None
    text_image_color_type: str = 'binary'

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
            from ocr4all_pixel_classifier.lib.dataset import Dataset, SingleData
            return Dataset(
                [SingleData(image=d.line_image if self.params.image_input == ImageInput.LINE_IMAGE else d.region,
                            binary=((d.line_image < 125) * 255).astype(np.uint8),
                            mask=d.mask,
                            line_height_px=self.params.origin_staff_line_distance if self.params.origin_staff_line_distance is not None else d.operation.page.avg_staff_line_distance(),
                            original_shape=d.line_image.shape,
                            user_data=d) for d in self.load(callback)], {})
        else:
            raise NotImplementedError()

    def to_line_detection_dataset(self, callback: Optional[DatasetCallback] = None) -> List[RegionLineMaskData]:
        return self.load(callback)

    def to_memory_dataset(self, callback: Optional[DatasetCallback] = None):
        if self.params.origin_staff_line_distance == self.params.target_staff_line_distance:
            images = []
            masks = []
            data = []
            # import matplotlib.pyplot as plt
            # cmap = plt.get_cmap('Set1')

            for ind, x in enumerate(self.load(callback)):
                # from PIL import Image
                # rgba_img = cmap(x.mask)
                # rgb_img = np.delete(rgba_img, 3, 2)*255
                # rgb_img = rgb_img.astype(np.uint8)
                # backgorund = Image.fromarray(x.region).convert("RGBA")
                # overlay= Image.fromarray(rgb_img).convert("RGBA")
                # overlay.save(str(ind)+"overlay.png")
                # new_image = Image.blend(backgorund, overlay, 0.5)
                # new_image.save(str(ind)+".png")
                images.append(x.line_image if self.params.image_input == ImageInput.LINE_IMAGE else x.region)
                masks.append(x.mask)
                data.append(x)
            df = pd.DataFrame(data={'images': images, 'masks': masks, 'original': data})
            return df
        else:
            raise NotImplementedError()
            pass
    def to_drop_capital_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
        from omr.steps.layout.drop_capitals.torch_dataset import DropCapitalDataset
        d = self.load(callback)
        images = []
        masks = []
        additional_data = []

        for instance in d:
            images.append(instance.operation.images[0].image)
            masks.append(instance.operation.images[1].image)
            if not train:
                additional_data.append(instance)
            #        from matplotlib import pyplot as plt
            #f, axarr = plt.subplots(2, 1)
            #axarr[0].imshow(instance.operation.images[0].image)
            #axarr[1].imshow(instance.operation.images[1].image)

            #plt.show()

        return DropCapitalDataset(imgs=images, masks=masks, additional_data=additional_data)
    def to_calamari_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
        from calamari_ocr.ocr.datasets.dataset import RawDataSet, DataSetMode
        marked_symbols = self.load(callback)

        def get_input_image(d: RegionLineMaskData):
            if self.params.cut_region:
                return d.line_image
            elif self.params.masks_as_input:
                hot = d.operation.images[3].image
                hot[:, :, 0] = (d.line_image if self.params.cut_region else d.region)
                return hot.transpose([1, 0, 2])
            else:
                return d.region

        images = [get_input_image(d).astype(np.uint8) for d in marked_symbols]
        from PIL import Image
        for ind, i in enumerate(images):
            Image.fromarray(i).save("/tmp/images/image_" + str(ind) + ".png")
        if self.params.neume_types_only:
            gts = [d.calamari_sequence(self.params.calamari_codec).calamari_neume_types_str for d in marked_symbols]
        else:
            gts = [d.calamari_sequence(self.params.calamari_codec).calamari_str for d in marked_symbols]
        return RawDataSet(DataSetMode.TRAIN if train else DataSetMode.PREDICT, images=images, texts=gts)

    def to_text_line_calamari_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
        from omr.steps.text.calamari.calamari_interface import RawData

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
        #print(gts)
        #import uuid
        #uuid = str(uuid.uuid4()) + "_"
        #path = "/tmp/images/"
        #if not os.path.exists(path):
        #    os.mkdir(path)
        #from PIL import Image
        #import csv
        #with open(os.path.join(path, "labels.csv"), 'w', newline='') as csvfile:
        #    fieldnames = ['filename', 'words']
        #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #    writer.writeheader()
        #    for ind, (i, t) in enumerate(zip(images, gts)):
        #        with open(os.path.join(path, uuid + str(ind) + ".gt.txt"), 'w') as f:
        #            f.write(t)
        #        Image.fromarray(i).save(os.path.join(path, uuid + str(ind) + ".png"))
        #        writer.writerow({'filename': uuid+ str(ind) + ".png", 'words': t})
        #print("123")
        #exit()
        return RawData(images=images, gt_files=gts)

    def to_nautilus_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
        marked_symbols = self.load(callback)
        import tempfile
        import shutil
        def get_input_image(d: RegionLineMaskData):
            if self.params.cut_region:
                return d.line_image
            elif self.params.masks_as_input:
                hot = d.operation.images[3].image
                hot[:, :, 0] = (d.line_image if self.params.cut_region else d.region)
                return hot.transpose([1, 0, 2])
            else:
                return d.region

        images = [get_input_image(d).astype(np.uint8) for d in marked_symbols]
        from PIL import Image
        #for ind, i in enumerate(images):
        #    Image.fromarray(i).save("/tmp/images/image_" + str(ind) + ".png")
        if self.params.neume_types_only:
            gts = [d.calamari_sequence(self.params.calamari_codec).calamari_neume_types_str for d in marked_symbols]
        else:
            gts = [d.calamari_sequence(self.params.calamari_codec).calamari_str for d in marked_symbols]
        """
        if train:
            path = "/tmp/train/"
        else:
            path = "/tmp/val/"

        if not os.path.exists(path):
            os.mkdir(path)
        from PIL import Image
        import csv
        import uuid
        uuid = str(uuid.uuid4()) + "_"
        with open(os.path.join(path, "labels.csv"), 'w', newline='') as csvfile:
            fieldnames = ['filename', 'words']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ind, (i, t) in enumerate(zip(images, gts)):
                with open(os.path.join(path, uuid + str(ind) + ".gt.txt"), 'w') as f:
                    f.write(t)
                Image.fromarray(i).save(os.path.join(path, uuid + str(ind) + ".png"))
                writer.writerow({'filename': uuid+ str(ind) + ".png", 'words': t})
        """
        return images, gts

    def to_text_line_nautilus_dataset(self, train=False, callback: Optional[DatasetCallback] = None):
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

        return (images, gts)
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

                logger.info("Exception during processing of page: {}".format(f.page.location.local_path()))
