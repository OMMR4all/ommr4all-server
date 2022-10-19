from database.start_up.load_text_variants_in_memory import load_model
from django.apps import AppConfig


class LyricsConfig(AppConfig):
    name = 'database.lyric_database'
    def ready(self):
        load_model()