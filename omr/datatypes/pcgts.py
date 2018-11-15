from json import JSONDecodeError
from typing import Generator, Tuple

from omr.datatypes import Meta, Page, MusicLine, MusicRegion

import numpy as np
import main.book as book
import os

from PIL import Image


class PcGts:
    def __init__(self, meta: Meta, page: Page):
        self.meta: Meta = meta
        self.page: Page = page

    @staticmethod
    def from_file(file: book.File):
        filename = file.local_path()
        if filename.endswith(".json"):
            import json
            with open(filename, 'r') as f:
                pcgts = PcGts.from_json(json.load(f), file.page)
        else:
            raise Exception("Invalid file extension of file '{}'".format(filename))

        if len(pcgts.page.image_filename) == 0:
            from main.book import file_definitions
            pcgts.page.image_filename = file_definitions['color_deskewed'].output[0]

        img_path = os.path.join(os.path.split(filename)[0], pcgts.page.image_filename)
        if not os.path.exists(img_path):
            raise Exception('Missing image file at {}'.format(img_path))

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
    def from_json(json: dict, location: book.Page):
        pcgts = PcGts(
            Meta.from_json(json.get('meta', {})),
            Page.from_json(json.get('page', {}), location=location),
        )
        pcgts.page.image_width, pcgts.page.image_height = Image.open(location.file('color_deskewed', True).local_path()).size
        return pcgts

    def to_json(self):
        return {
            'meta': self.meta.to_json(),
            'page': self.page.to_json(),
        }


if __name__ == '__main__':
    from omr.datatypes import *
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




