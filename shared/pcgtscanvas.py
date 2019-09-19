import numpy as np
from typing import Union, Iterable
from PIL import Image
import cv2

from database.file_formats.pcgts import MusicSymbol, StaffLine, Page, PageScaleReference, StaffLines, SymbolType, \
    GraphicalConnectionType, Rect
from database.file_formats.pcgts.page import Annotations, ClefType


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

    def draw(self, elem: Union[MusicSymbol, StaffLine, Iterable, Annotations]) -> 'PcGtsCanvas':
        from omr.steps.text.predictor import SingleLinePredictionResult as TextPredictionResult
        from omr.steps.syllables.syllablesfromtext.predictor import MatchResult
        from omr.steps.syllables.predictor import PredictionResult as SyllablesPredictionResult

        def scale(x):
            if isinstance(x, Rect):
                return Rect(scale(x.origin), scale(x.size))
            return np.round(self.page.page_to_image_scale(x, self.scale_reference)).astype(int)

        if isinstance(elem, MusicSymbol):
            cv2.circle(self.img, tuple(scale(elem.coord.p)), self.avg_line_distance // 8, color=self._color_for_music_symbol(elem), thickness=-1)
        elif isinstance(elem, StaffLine):
            sl: StaffLine = elem
            sl.draw(self.img, thickness=self.avg_line_distance // 10, scale=scale)
        elif isinstance(elem, TextPredictionResult):
            r: TextPredictionResult = elem
            aabb = scale(r.line.operation.text_line.aabb)
            t, b = int(aabb.top()), int(aabb.bottom())
            for text, pos in r.text:
                pos = scale(pos)
                self.img[t:b, int(pos)] = (255, 0, 0)
                cv2.putText(self.img, text, (int(pos), b - 5), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=4, color=(255, 0, 0), thickness=2)
        elif isinstance(elem, MatchResult):
            r: MatchResult = elem
            aabb = scale(r.text_line.aabb)
            t, b = int(aabb.top()), int(aabb.bottom())
            for syl in r.syllables:
                pos = int(scale(syl.xpos))
                self.img[t:b, pos] = (255, 0, 0)
                text = syl.syllable.text.replace(" ", "_")
                cv2.putText(self.img, text, (pos - 20, b + 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(205, 20, 100), thickness=2)
        elif isinstance(elem, SyllablesPredictionResult):
            r: SyllablesPredictionResult = elem
            self.draw(r.annotations)
        elif isinstance(elem, Annotations):
            for c in elem.connections:
                aabb = scale(c.text_region.aabb)
                t, b = int(aabb.top()), int(aabb.bottom())
                for sc in c.syllable_connections:
                    x = int(scale(sc.note.coord.x))
                    self.img[t:b, x] = (0, 255, 0)
                    text = sc.syllable.text.replace(" ", "_")
                    cv2.putText(self.img, text, (int(x) - 20, b + 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(20, 205, 100), thickness=2)

        elif isinstance(elem, Iterable):
            for e in elem:
                self.draw(e)

        return self

    def _color_for_music_symbol(self, ms: MusicSymbol):
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

        return 255, 255, 255

    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.img)
        plt.show()
