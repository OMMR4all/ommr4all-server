import os
from typing import List

from loguru import logger

from database import DatabaseBook
# Adjust this import based on your actual project structure
from database.database_page import DatabasePage


def shift_staff_lines_by_pixels(pages: List['DatabasePage'], pixel_delta_y: float):
    """
    Moves all staff lines in the provided pages by a set number of pixels,
    leveraging the Page class's built-in scaling methods.

    Args:
        pages: A list of DatabasePage objects.
        pixel_delta_y: The number of pixels to move the staff lines.
                       Positive values move lines DOWN.
                       Negative values move lines UP.
    """
    for db_page in pages:
        try:
            pcgts_obj = db_page.pcgts(create_if_not_existing=False)
        except Exception as e:
            logger.error(f"Failed to load PcGts for {db_page.page}: {e}")
            continue

        if not pcgts_obj:
            logger.warning(f"Skipping {db_page.page}: No PcGts data found.")
            continue

        page = pcgts_obj.page

        for music_line in page.all_music_lines():
            for staff_line in music_line.staff_lines:
                # 1. Scale up: Convert relative Coords to a new absolute pixel Coords object
                pixel_coords = page.page_to_image_scale(staff_line.coords)

                # 2. Shift: Apply the pixel delta directly to the Y-axis points
                pixel_coords.points[:, 1] += pixel_delta_y

                # 3. Scale down: Convert back to relative Coords and reassign
                staff_line.coords = page.image_to_page_scale(pixel_coords)

                # 4. Refresh internal caches (_center_y and _dewarped_y)
                staff_line.update()

        # Save the modifications back to the local JSON file
        try:
            pcgts_file_path = db_page.file('pcgts').local_path()
            pcgts_obj.to_file(pcgts_file_path)
            logger.info(f"Successfully shifted staff lines by {pixel_delta_y}px on page: {db_page.page}")
        except Exception as e:
            logger.error(f"Error saving page {db_page.page}: {e}")

# --- Example Usage ---
# Assuming you have a list of pages from a book:
# my_pages = my_database_book.pages()
# shift_staff_lines(my_pages, delta_y=-10.5) # Moves all staff lines UP by 10.5 pixels

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    book = DatabaseBook("LondonBLRoyal2BIV")
    pages = book.pages()
    #pages = [p for p in pages if p.page == "00006r"]
    #for p in pages:
    #    print(p.page)
    shift_staff_lines_by_pixels(pages, pixel_delta_y=-6) # Moves all staff lines UP by 10.5 pixels
