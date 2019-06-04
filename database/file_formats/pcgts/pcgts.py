from database.file_formats.pcgts.meta import Meta
from database.file_formats.pcgts.page import Page
from database import DatabaseFile, DatabasePage
import os

from PIL import Image


class PcGts:
    def __init__(self, meta: Meta, page: Page):
        self.meta: Meta = meta
        self.page: Page = page

    @staticmethod
    def from_file(file: DatabaseFile):
        filename = file.local_path()
        if filename.endswith(".json"):
            import json
            with open(filename, 'r') as f:
                pcgts = PcGts.from_json(json.load(f), file.page)
        else:
            raise Exception("Invalid file extension of file '{}'".format(filename))

        if len(pcgts.page.image_filename) == 0:
            pcgts.page.image_filename = DatabaseFile.file_definitions()['color_deskewed'].output[0]

        return pcgts

    def to_file(self, filename):
        if filename.endswith(".json"):
            import json
            # first dump to keep file if an error occurs
            s = json.dumps(self.to_json(), indent=2)
            with open(filename, 'w') as f:
                f.write(s)
        else:
            raise Exception("Invalid file extension of file '{}'".format(filename))

    @staticmethod
    def from_json(json: dict, location: DatabasePage):
        # original an deskewed have same shape
        image_shape = Image.open(location.file('color_original', True).local_path()).size
        pcgts = PcGts(
            Meta.from_json(json.get('meta', {})),
            Page.from_json(json.get('page', {}), location=location),
        )
        pcgts.page.image_width, pcgts.page.image_height = image_shape
        return pcgts

    def to_json(self):
        return {
            'meta': self.meta.to_json(),
            'page': self.page.to_json(),
        }


if __name__ == '__main__':
    from database.file_formats.pcgts import TextRegion, TextRegionType, Page, Meta
    pcgts = PcGts(Meta(), Page(
        [
            TextRegion(
                '1',
                TextRegionType.LYRICS
            )
        ]
    ))

    print(pcgts.to_json())
    print(PcGts.from_json(pcgts.to_json()).to_json() == pcgts.to_json())




