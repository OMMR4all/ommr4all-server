from typing import List, TYPE_CHECKING
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from database.file_formats.pcgts.page import Page
    from database.file_formats.pcgts.page.textline import TextLine

class ReadingOrder:
    def __init__(self, page: 'Page', reading_order: List['TextLine'] = None):
        self.page = page
        self.reading_order = reading_order if reading_order else []

    @staticmethod
    def from_json(d: dict, page: 'Page'):
        lines = []
        for id in d.get('lyricsReadingOrder', []):
            line = page.text_line_by_id(id)
            if not line:
                logger.warning("TextLine with id {} not found".format(id))
            else:
                lines.append(line)

        return ReadingOrder(page, lines)

    def to_json(self):
        return {
            'lyricsReadingOrder': [l.id for l in self.reading_order]
        }
