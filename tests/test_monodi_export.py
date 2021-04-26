import unittest

import ommr4all.settings as settings
from database import DatabaseBook
import os
import sys
import logging
import json
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
from shared.jsonparsing import drop_all_attributes

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')


class TestMonodiExport(unittest.TestCase):
    def test_single_line_001(self):
        try:
            book = DatabaseBook('demo')
            file = book.page("page_test_monodi_export_001")
            pcgts = file.pcgts()
            root = PcgtsToMonodiConverter([pcgts]).root
            j = root.to_json()
            drop_all_attributes(j, 'uuid')

            with open(file.local_file_path('monodi.json'), 'r') as f:
                ref = json.load(f)

            self.maxDiff = None
            #self.assertEqual(ref, j)
        except Exception as e:
            logging.exception(e)
            raise e

