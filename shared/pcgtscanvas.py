import numpy as np
from typing import Union, Iterable
from PIL import Image, ImageFont, ImageDraw
import cv2

from database.file_formats.pcgts import MusicSymbol, StaffLine, Page, PageScaleReference, StaffLines, SymbolType, \
    GraphicalConnectionType, Rect
from database.file_formats.pcgts.page import Annotations, ClefType


class PcGtsCanvas:
    def __init__(self, page: Page, scale_reference: PageScaleReference, no_background=False, file='color'):
        self.page = page
        self.scale_reference = scale_reference
        self.img = np.array(Image.open(page.location.file(scale_reference.file(file)).local_path()))
        self.avg_line_distance = int(page.page_to_image_scale(page.avg_staff_line_distance(), scale_reference))
        self.font = ImageFont.truetype('/usr/share/fonts/TTF/NotoSansNerdFont-Light.ttf', 40)

        if no_background:
            self.img = np.full(self.img.shape, 255, dtype=self.img.dtype)

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

    def draw(self, elem: Union[MusicSymbol, StaffLine, Iterable, Annotations], **kwargs) -> 'PcGtsCanvas':
        from omr.steps.text.predictor import SingleLinePredictionResult as TextPredictionResult
        from omr.steps.syllables.syllablesfromtext.predictor import MatchResult
        from omr.steps.syllables.predictor import PredictionResult as SyllablesPredictionResult

        fac = kwargs.get('scale', 1)

        def scale(x):
            if isinstance(x, Rect):
                return Rect(scale(x.origin), scale(x.size))
            return np.round(self.page.page_to_image_scale(x, self.scale_reference)).astype(int)

        if isinstance(elem, Rect):
            r: Rect = elem
            self.img[r.top():r.bottom(), r.left():r.right()] = kwargs.get('color', (255, 255, 255))
        elif isinstance(elem, MusicSymbol):
            color = self.__class__.color_for_music_symbol(elem)
            if kwargs.get('invert', False):
                color = tuple(map(int, 255 - np.array(color, dtype=int)))
            pos = tuple(scale(elem.coord.p))
            cv2.circle(self.img, pos, int(self.avg_line_distance / 8 * fac), color=color, thickness=-1)
            if elem.symbol_type == elem.symbol_type.CLEF:
                if elem.clef_type == elem.clef_type.C:
                    self.img[int(pos[1]-self.avg_line_distance  * 0.8):int(pos[1]+self.avg_line_distance * 0.8), int(pos[0]-self.avg_line_distance  * 0.3):int(pos[0]+self.avg_line_distance  * 0.3)] = color
                else:
                    self.img[int(pos[1]-self.avg_line_distance  * 0.8):int(pos[1]+self.avg_line_distance * 0.8), int(pos[0]-self.avg_line_distance  * 0.4):int(pos[0]+self.avg_line_distance  * 0.4)] = color


            else:
                self.img[int(pos[1]-self.avg_line_distance / 4 * 0.8):int(pos[1]+self.avg_line_distance / 4 * 0.8), int(pos[0]-self.avg_line_distance / 4 * 0.8):int(pos[0]+self.avg_line_distance / 4 * 0.8)] = color

        elif isinstance(elem, StaffLine):
            sl: StaffLine = elem
            sl.draw(self.img, thickness=self.avg_line_distance // 10, scale=scale)
        elif isinstance(elem, TextPredictionResult):
            r: TextPredictionResult = elem
            aabb = scale(r.line.operation.text_line.aabb)
            t, b = int(aabb.top()), int(aabb.bottom())
            color = kwargs.get('color', (255, 0, 0))
            if kwargs.get('background', False):
                rect = aabb.copy()
                rect.origin.p[1] += rect.size.h
                self.draw(rect, color=(255, 255, 255))

            canvas = Image.fromarray(self.img)
            draw = ImageDraw.Draw(canvas)
            for text, pos in r.text:
                pos = scale(pos)
                draw.line(((int(pos), t), (int(pos), b)), color)
                draw.text((int(pos), b - 5), text, font=self.font, fill=color)

            self.img = np.array(canvas)
        elif isinstance(elem, MatchResult):
            r: MatchResult = elem
            aabb = scale(r.text_line.aabb)
            t, b = int(aabb.top()), int(aabb.bottom())
            for syl in r.syllables:
                pos = int(scale(syl.xpos))
                self.img[t:b, pos] = (255, 0, 0)
                text = syl.syllable.text.replace(" ", "_")
                cv2.putText(self.img, text, (pos , b ), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(205, 20, 100), thickness=2)
        elif isinstance(elem, SyllablesPredictionResult):
            r: SyllablesPredictionResult = elem
            self.draw(r.annotations, **kwargs)
        elif isinstance(elem, Annotations):
            for c in elem.connections:
                aabb = scale(c.text_region.aabb)
                t, b = int(aabb.top()), int(aabb.bottom())
                for sc in c.syllable_connections:
                    x = int(scale(sc.note.coord.x))
                    self.img[t:b, x] = (0, 255, 0)
                    text = sc.syllable.text.replace(" ", "_")
                    cv2.putText(self.img, text, (int(x) - 20, b + 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(20, 205, 100), thickness=2)

        elif isinstance(elem, Iterable):
            for e in elem:
                self.draw(e, **kwargs)

        return self

    @classmethod
    def color_for_music_symbol(cls, ms: MusicSymbol, inverted=False, default_color=(255, 255, 255)):
        def wrapper():
            if ms is None:
                return default_color

            if ms.symbol_type == SymbolType.CLEF:
                if ms.clef_type == ClefType.C:
                    return 255, 50, 50
                elif ms.clef_type == ClefType.F:
                    return 160, 0, 0

            elif ms.symbol_type == SymbolType.NOTE:
                if ms.graphical_connection == GraphicalConnectionType.NEUME_START:
                    return 0, 100, 255
                elif ms.graphical_connection == GraphicalConnectionType.LOOPED:
                    return 0, 0, 255
                elif ms.graphical_connection == GraphicalConnectionType.GAPED:
                    return 100, 0, 255

            elif ms.symbol_type == SymbolType.ACCID:
                return 0, 255, 0

            return default_color

        color = wrapper()
        if inverted:
            return tuple([255 - v for v in color])

        return color

    def show(self):
        import matplotlib.pyplot as plt
        self.render_to_ax(plt)
        plt.show()

    def render_to_ax(self, ax):
        if len(self.img.shape) == 3:
            ax.imshow(self.img)
        else:
            ax.imshow(self.img, cmap='gray')

    @staticmethod
    def show_all(canvases: Iterable['PcGtsCanvas'], hor=True):
        import matplotlib.pyplot as plt
        canvases = list(canvases)
        if hor:
            f, ax = plt.subplots(nrows=1, ncols=len(canvases), sharex='all', sharey='all')
        else:
            f, ax = plt.subplots(nrows=len(canvases), sharex='all', sharey='all')

        for a, c in zip(ax, canvases):
            c.render_to_ax(a)

        plt.show()
