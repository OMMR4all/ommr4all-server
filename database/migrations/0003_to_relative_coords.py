from django.db import migrations
from database import DatabaseBook
from database.file_formats import PcGts
import json
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


def pcgts_to_relative_coords(apps, schema_editor):
    books = DatabaseBook.list_available()
    for book in tqdm(books, "Converting to relative coords"):
        for page in book.pages():
            pcgts_file = page.file('pcgts')
            if not pcgts_file.exists():
                continue

            pcgts = PcGts.from_file(pcgts_file)
            pcgts.page.to_relative_coords()
            pcgts.to_file(pcgts_file.local_path())


def remove_invalid_files(apps, schema_editor):
    books = DatabaseBook.list_available()
    for book in tqdm(books, "Removing old files"):
        for page in book.pages():
            obsolete_files = [
                'annotation.json',
                'binary_cropped.png',
                'binary_cropped_preview.jpg',
                'binary_deskewed.png',
                'binary_deskewed_preview.jpg',
                'binary_original.png',
                'binary_original_preview.jpg',
                'color_cropped.jpg',
                'color_cropped_preview.jpg',
                'color_deskewed.jpg',
                'color_deskewed_preview.jpg',
                'gray_cropped.jpg',
                'gray_cropped_preview.jpg',
                'gray_deskewed.jpg',
                'gray_deskewed_preview.jpg',
                'gray_original.jpg',
                'gray_original_preview.jpg',
                'connected_components_deskewed.pkl',
            ]
            for f in obsolete_files:
                f = page.local_file_path(f)
                if os.path.exists(f):
                    os.remove(f)


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0002_remove_words_and_neume_connector'),
    ]

    operations = [
        migrations.RunPython(pcgts_to_relative_coords),
        migrations.RunPython(remove_invalid_files),
    ]
