"""Every test run uses the fixture storage, never the real books.

Creating the test database runs data migrations (e.g. database/migrations/0002_*) that rewrite
the pcgts.json of every book in the storage root. Test modules used to redirect the storage
themselves, so a run of only modules without that redirect (e.g. tests.test_symbol_classes)
rewrote the user's books. The test runner imports this package before any test module and
before it sets up the database, so the redirect here holds for every run.
"""
import os

from django.conf import settings as django_settings

import ommr4all.settings as settings

_FIXTURE_STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
# the environment variable reaches subprocesses that re-import the settings; the database code
# reads the settings module itself (`import ommr4all.settings as settings`), not django.conf
os.environ['OMMR4ALL_STORAGE_ROOT'] = _FIXTURE_STORAGE
settings.PRIVATE_MEDIA_ROOT = _FIXTURE_STORAGE
django_settings.PRIVATE_MEDIA_ROOT = _FIXTURE_STORAGE
