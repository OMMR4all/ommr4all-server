import os
from PIL import Image
import imagehash

DRY_RUN = False

FOLDER_1_PATH = r'/tmp/images/'
FOLDER_2_PAGES_PATH = r'/tmp/Ass695/pages/'

TOLERANCE_THRESHOLD = 4



def get_image_data(image_path):
    """
    Returns a tuple: (Aspect Ratio, Perceptual Hash)
    """
    try:
        with Image.open(image_path) as img:
            aspect_ratio = round(img.width / img.height, 2)
            img_hash = imagehash.phash(img)

            return aspect_ratio, img_hash
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None


def main():
    print("--- Starting Precision Matching ---")
    if DRY_RUN:
        print("DRY RUN MODE: No files will be changed")

    print(f"Scanning Reference Folder: {FOLDER_1_PATH}...")
    folder1_data = []

    for filename in os.listdir(FOLDER_1_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(FOLDER_1_PATH, filename)

            # Get Aspect Ratio AND Hash
            ar, h = get_image_data(full_path)

            if h:
                name_no_ext = os.path.splitext(filename)[0]
                folder1_data.append({'name': name_no_ext, 'hash': h, 'ar': ar})

    print(f"Loaded {len(folder1_data)} reference images.\n")

    # 2. Iterate Subfolders
    print(f"Scanning Pages Folder: {FOLDER_2_PAGES_PATH}...")

    for subfolder_name in os.listdir(FOLDER_2_PAGES_PATH):
        subfolder_path = os.path.join(FOLDER_2_PAGES_PATH, subfolder_name)
        print(f"subfolder_path {subfolder_path}")
        if not os.path.isdir(subfolder_path):
            continue

        target_image = os.path.join(subfolder_path, "color_original.jpg")

        target_ar, target_hash = get_image_data(target_image)
        if not target_hash:
            continue

        best_match_name = None
        best_diff = 100

        for ref in folder1_data:
            if ref['ar'] != target_ar:
                continue

            diff = ref['hash'] - target_hash

            if diff < best_diff:
                best_diff = diff
                best_match_name = ref['name']

        if best_match_name and best_diff <= TOLERANCE_THRESHOLD:
            new_subfolder_path = os.path.join(FOLDER_2_PAGES_PATH, best_match_name)

            if subfolder_name == best_match_name:
                print(f"{subfolder_name} is correct.")
                continue

            if os.path.exists(new_subfolder_path):
                print(f"Target folder '{best_match_name}' already exists. Skipping '{subfolder_name}'")
                continue

            print(f"[MATCH] Diff: {best_diff} | AR: {target_ar}")
            print(f"       Renaming '{subfolder_name}' -> '{best_match_name}'")

            if not DRY_RUN:
                try:
                    os.rename(subfolder_path, new_subfolder_path)
                except OSError as e:
                    print(f"       Error renaming: {e}")
        else:
            print(f"No match for '{subfolder_name}' (Best diff: {best_diff})")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()