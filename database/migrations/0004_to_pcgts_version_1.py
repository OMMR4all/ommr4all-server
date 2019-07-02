from django.db import migrations
from database import DatabaseBook
from database.file_formats.pcgts.jsonloader import update_pcgts
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def pcgts_update_version(apps, schema_editor):
    books = DatabaseBook.list_available()
    version = 1
    for book in tqdm(books, "Converting to pcgts version {}".format(version)):
        for page in book.pages():
            pcgts_file = page.file('pcgts')
            if not pcgts_file.exists():
                continue

            with open(pcgts_file.local_path()) as f:
                j = json.load(f)

            upgraded = update_pcgts(j, target_version=version)

            if upgraded:
                with open(pcgts_file.local_path(), 'w') as f:
                    json.dump(j, f, indent=2)


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0003_to_relative_coords'),
    ]

    operations = [
        migrations.RunPython(pcgts_update_version),
    ]
