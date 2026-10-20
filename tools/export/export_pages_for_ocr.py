import os
from random import shuffle

from database import DatabaseBook, DatabasePage
from database.file_formats.performance.pageprogress import Locks
import shutil
import os
import uuid


def save_image_with_uuid(image_source_path, string_list, output_directory):
    """
    Copies an image to a directory with a random UUID name and
    saves a matching text file with the same UUID.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    unique_id = str(uuid.uuid4())

    extension = os.path.splitext(image_source_path)[1]

    new_image_name = unique_id + extension
    new_text_name = unique_id + ".txt"

    image_dest_path = os.path.join(output_directory, new_image_name)
    text_dest_path = os.path.join(output_directory, new_text_name)

    shutil.copy2(image_source_path, image_dest_path)
    string_list = [i.replace("-","") for i in string_list]
    with open(text_dest_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(string_list))

    print(f"Saved: {new_image_name} and {new_text_name}")
    return unique_id

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    # get all books from database
    books = [DatabaseBook("Graduel_Part_1_gt"), DatabaseBook("Graduel_Part_2_gt"), DatabaseBook("Graduel_Part_3_gt"),]
    books = [DatabaseBook("Geesebook1_complete_fixed_ro")]
    books = [DatabaseBook("mul_2_rsync_gt2")]
    books = [DatabaseBook("Koeln_Dombibl_1001b_part_gt")]
    books = [DatabaseBook("Pa_14819_gt")]

    excepted_ids = []
    pages_ad = []

    for b in books:

        pages = b.pages()

        for page in pages:
            page_id = page.page
            annotation = page.pcgts().page.annotations
            c_progress = page.page_progress()

            if Locks.TEXT in c_progress.locked and  c_progress.locked[Locks.TEXT] == True:
                ro = page.pcgts().page.reading_order.reading_order
                ro = [i for i in ro if i]
                lyric = [i.text() for i in ro]

                pages_ad.append(page)

    shuffle(pages_ad)

    for page in pages_ad[:20]:
        page: DatabasePage = page
        text_lines = page.pcgts().page.all_text_lines(only_lyric=True)
        ro = page.pcgts().page.reading_order.reading_order
        lyric = [i.text(with_drop_capital = False) for i in ro]
        pcgts = page.pcgts()
        page = pcgts.dataset_page()

        path = page.file("color_original").local_path()
        save_image_with_uuid(page.file("color_original").local_path(), lyric, "/tmp/rese/Pa14819/")



