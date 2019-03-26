from django.db import migrations
from .. import DatabaseBook
import json


def convert_page_process(apps, schema_editor):
    books = DatabaseBook.list_available()
    for book in books:
        for page in book.pages():
            page_progress = page.file('page_progress')
            if not page_progress.exists():
                continue

            with open(page_progress.local_path(), 'r') as f:
                progress = json.load(f)

            if 'locked' not in progress:
                continue

            out_locks = {}
            locks = progress['locked']
            out_locks['StaffLines'] = any([locks.get('StaffLines', False), locks.get('CreateStaffLines', False), locks.get('GroupStaffLines', False), locks.get('SplitStaffLines', False)])
            out_locks['Layout'] = any([locks.get('Layout', False), locks.get('LayoutExtractConnectedComponents', False), locks.get('LayoutLassoArea', False)])
            out_locks['Symbols'] = any([locks.get('Symbols', False), locks.get('Symbol', False)])
            out_locks['Text'] = any([locks.get('Text', False), locks.get('Lyrics', False), locks.get('Syllables', False)])

            progress['locked'] = out_locks

            with open(page_progress.local_path(), 'w') as f:
                json.dump(progress, f, indent=2)


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.RunPython(convert_page_process),
    ]
