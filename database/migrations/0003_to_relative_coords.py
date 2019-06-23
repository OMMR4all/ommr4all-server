from django.db import migrations
from database import DatabaseBook
import json
import logging
from tqdm import tqdm
import os
from PIL import Image

logger = logging.getLogger(__name__)


def to_relative_coords(d: dict, size):
    w, h = size
    all_coords = []

    if d.get('relative_coords', False):
        # already in relative coords
        return

    def p2p(p: str) -> str:
        global all_coords
        x, y = tuple(map(float, p.split(',')))
        all_coords += x, y
        return "{},{}".format(x / h, y / h)

    def c2p(c: str) -> str:
        return ' '.join([p2p(p) for p in c.split(' ')])

    for r in d.get('text_regions', []):
        r['coords'] = c2p(r['coords'])
        for l in r.text_lines:
            l['coords'] = c2p(l['coords'])

    for r in d.get('music_regions', []):
        r['coords'] = c2p(r['coords'])
        for l in r.get('staffs', []):
            l['coords'] = c2p(l['coords'])
            for s in l.get('staffLines', []):
                s['coords'] = c2p(s['coords'])

            for s in l.get('symbols', []):
                s['coord'] = p2p(s['coord'])
                if s['type'] == 0:
                    for nc in s['notes']:
                        nc['coord'] = p2p(nc['coord'])

    was_local = all([0 <= p <= 1 for p in all_coords])
    return was_local


def pcgts_to_relative_coords(apps, schema_editor):
    books = DatabaseBook.list_available()
    for book in tqdm(books, "Converting to relative coords"):
        for page in book.pages():
            pcgts_file = page.file('pcgts')
            size = Image.open(page.file('color_original').local_path()).size
            if not pcgts_file.exists():
                continue

            with open(pcgts_file.local_path()) as f:
                j = json.load(f)
            was_local = to_relative_coords(j, size)
            if not was_local:
                with open(pcgts_file.local_path(), 'w') as f:
                    json.dump(j, f)


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
