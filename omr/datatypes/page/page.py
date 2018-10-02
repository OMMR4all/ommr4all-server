from omr.datatypes.page import TextRegion, MusicRegion
from typing import List


class Page:
    def __init__(self,
                 text_regions: List[TextRegion]=list(),
                 music_regions: List[MusicRegion]=list(),
                 image_filename="", image_height=0, image_width=0):
        self.text_regions = text_regions
        self.music_regions = music_regions
        self.image_filename = image_filename
        self.image_height = image_height
        self.image_width = image_width

    def syllable_by_id(self, syllable_id):
        for t in self.text_regions:
            r = t.syllable_by_id(syllable_id)
            if r:
                return r

        return None

    def _resolve_cross_refs(self):
        for t in self.text_regions:
            t._resolve_cross_refs(self)

        for m in self.music_regions:
            m._resolve_cross_refs(self)

    @staticmethod
    def from_json(json: dict):
        page = Page(
            [TextRegion.from_json(t) for t in json.get('textRegions', [])],
            [MusicRegion.from_json(m) for m in json.get('musicRegions', [])],
            json.get('imageFilename', ""),
            json.get('imageHeight', 0),
            json.get('imageWidth', 0),
        )
        page._resolve_cross_refs()
        return page

    def to_json(self):
        return {
            "textRegions": [t.to_json() for t in self.text_regions],
            "musicRegions": [m.to_json() for m in self.music_regions],
            "imageFilename": self.image_filename,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
        }
