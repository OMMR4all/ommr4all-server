import os

from django.apps import AppConfig

from database.start_up.load_text_variants_in_memory import lyrics_store


class LyricsConfig(AppConfig):
    name = 'database.lyric_database'

    def ready(self):
        lyrics_store.load()