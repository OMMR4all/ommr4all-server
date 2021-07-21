from django.db import migrations
from database import DatabaseBook
import json
import logging
import os
from ommr4all.settings import BASE_DIR

logger = logging.getLogger(__name__)


def populate_lyrics(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('database', '0008_fix_dataset_params'),
    ]

    operations = [
        migrations.RunPython(populate_lyrics),
    ]
