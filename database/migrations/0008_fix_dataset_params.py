from django.db import migrations
from database import DatabaseBook
import json
import logging
import os
from ommr4all.settings import DEFAULT_MODELS_ROOT, SHIPPED_DEFAULT_MODELS_ROOT

logger = logging.getLogger(__name__)


def fix_params(params):
    if not 'lyrics_normalization' in params:
        return False

    if isinstance(params['lyrics_normalization'], str):
        params['lyrics_normalization'] = {'lyrics_normalization': params['lyrics_normalization']}
    else:
        return False

    return True


def fix_file(path):
    if not os.path.exists(path):
        return

    with open(path) as f:
        params = json.load(f)

    if not fix_params(params):
        return

    with open(path, 'w') as f:
        json.dump(params, f, indent=2)


def fix_dataset_params(apps, schema_editor):
    # book models
    for book in DatabaseBook.list_available():
        if not os.path.exists(book.local_models_path()):
            continue

        for alg in os.listdir(book.local_models_path()):
            alg_dir = os.path.join(book.local_models_path(alg))
            for model in os.listdir(alg_dir):
                path = os.path.join(alg_dir, model, 'dataset_params.json')
                fix_file(path)

    # default models: the shipped ones and, when they were seeded onto a volume
    # (OMMR4ALL_DEFAULT_MODELS_ROOT), that copy as well. fix_file is idempotent.
    for default_models in {SHIPPED_DEFAULT_MODELS_ROOT, DEFAULT_MODELS_ROOT}:
        if not os.path.exists(default_models):
            continue
        for t in os.listdir(default_models):
            t_dir = os.path.join(default_models, t)
            for alg in os.listdir(t_dir):
                path = os.path.join(t_dir, alg, 'dataset_params.json')
                fix_file(path)


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0007_auto_20190910_1416'),
    ]

    operations = [
        migrations.RunPython(fix_dataset_params),
    ]
