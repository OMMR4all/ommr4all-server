import numpy as np
from typing import Union, Iterable
from PIL import Image
import cv2

from database.file_formats.pcgts import MusicSymbol, StaffLine, Page, PageScaleReference, StaffLines, SymbolType, \
    GraphicalConnectionType


class PcGtsCanvas:
    def __init__(self, page: Page, scale_reference: PageScaleReference):
        self.page = page
        self.scale_reference = scale_reference
        self.img = np.array(Image.open(page.location.file(scale_reference.file('color')).local_path()))
        self.avg_line_distance = int(page.page_to_image_scale(page.avg_staff_line_distance(), scale_reference))

    def draw_music_symbol_position_in_line(self, sl: StaffLines, s: MusicSymbol, color=(255, 0, 0), thickness=-1) -> 'PcGtsCanvas':
        c = sl.compute_coord_by_position_in_staff(s.coord.x, s.position_in_staff)
        def scale(x):
            return np.round(self.page.page_to_image_scale(x, self.scale_reference)).astype(int)
        if s.symbol_type == SymbolType.NOTE:
            if s.graphical_connection == GraphicalConnectionType.LOOPED:
                color = (np.array(color) // 4)
            elif s.graphical_connection == GraphicalConnectionType.GAPED:
                color = np.minimum(np.array(color) + 128, 255)

        cv2.circle(self.img, tuple(scale(c.p)), self.avg_line_distance // 5, color=tuple(map(int, color)), thickness=thickness)
        return self

    def draw(self, elem: Union[MusicSymbol, StaffLine, Iterable]) -> 'PcGtsCanvas':
        if isinstance(elem, Iterable):
            for e in elem:
                self.draw(e)

        def scale(x):
            return np.round(self.page.page_to_image_scale(x, self.scale_reference)).astype(int)

        if isinstance(elem, MusicSymbol):
            cv2.circle(self.img, tuple(scale(elem.coord.p)), self.avg_line_distance // 8, color=(255, 0, 0), thickness=-1)
        elif isinstance(elem, StaffLine):
            sl: StaffLine = elem
            sl.draw(self.img, thickness=self.avg_line_distance // 10, scale=scale)

        return self

    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.img)
        plt.show()
