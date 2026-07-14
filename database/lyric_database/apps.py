import os
import sys

from django.apps import AppConfig

from database.start_up.load_text_variants_in_memory import lyrics_store


class LyricsConfig(AppConfig):
    name = 'database.lyric_database'

    def ready(self):
        # Under `runserver`, Django's autoreloader runs two processes: a watcher
        # parent and the serving child. Both call django.setup(), so ready()
        # (and the heavy lyrics/syllable load) would run twice. The watcher
        # never serves requests, so skip the load there. RUN_MAIN is unset in
        # the watcher and 'true' in the child; it is also unset under
        # wsgi/mod_wsgi and other management commands, where 'runserver' is not
        # in argv, so those still load normally.
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            return
        lyrics_store.load()