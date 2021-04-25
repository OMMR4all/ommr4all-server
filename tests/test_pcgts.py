import unittest
import os
import logging
import ommr4all.settings as settings
from database.file_formats.pcgts.jsonloader import update_pcgts
from database.file_formats.pcgts import PcGts
import sys
import json
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')
raw_storage = os.path.join(BASE_DIR, 'tests', 'raw_storage')


class GenericTests(unittest.TestCase):
    def test_upgrade(self):
        with open(os.path.join(raw_storage, 'page_test_upgrade_001', 'pcgts.json')) as f:
            json0 = json.load(f)

        json1 = deepcopy(json0)
        self.assertTrue(update_pcgts(json1))
        self.maxDiff = None
        #self.assertEqual(json1, PcGts.from_json(json1, None).to_json())
