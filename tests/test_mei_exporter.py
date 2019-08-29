import unittest

from six import BytesIO

import ommr4all.settings as settings
from database import DatabaseBook
import os
import sys
import logging
from database.file_formats.exporter.mei.pcgts_to_mei4_exporter import PcgtsToMeiConverter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')


class TestMEIExport(unittest.TestCase):
    def test_single_line_001(self):
        try:
            book = DatabaseBook('demo')
            file = book.page("page_test_monodi_export_001")
            pcgts = file.pcgts()
            root = PcgtsToMeiConverter(pcgts)
            self.assertTrue(root.is_valid)

            # test to string and write
            root.to_string()
            buffer = BytesIO()
            root.write(buffer, pretty_print=True)

        except Exception as e:
            logging.exception(e)
            raise e

