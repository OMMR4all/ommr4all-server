import logging
import sys
import os
from unittest import TestCase

import ommr4all.settings as settings
from database import DatabaseBook
from database.file_formats.performance import LockState
from database.file_formats.performance.pageprogress import Locks
from restapi.operationworker.taskrunners.pageselection import PageSelection, PageSelectionParams, PageCount

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')


class TestBookOperations(TestCase):
    def test_page_selection(self):
        book = DatabaseBook('demo')
        p = PageSelectionParams(
            count=PageCount.ALL,
        )
        sel = PageSelection.from_params(p, book)
        self.assertListEqual([p.local_path() for p in sel.get_pages()], [p.local_path() for p in book.pages()])

    def test_pages_with_lock(self):
        book = DatabaseBook('demo')
        pages = book.pages_with_lock([LockState(Locks.STAFF_LINES, True)])
        self.assertListEqual([p.local_path() for p in pages], [book.page('page_test_lock').local_path()])

        pages = book.pages_with_lock([LockState(Locks.STAFF_LINES, False), LockState(Locks.SYMBOLS, True)])
        self.assertListEqual([p.local_path() for p in pages], [])




