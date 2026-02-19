import os
from PIL import Image
from pathlib import Path


def get_max_scaled_width(image_folder, target_height=960):
    max_w = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    folder_path = Path(image_folder)

    image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print("Keine Bilder gefunden!")
        return 0

    print(f"Prüfe {len(image_files)} Bilder...")

    for img_path in image_files:
        try:

            with Image.open(img_path) as img:
                orig_w, orig_h = img.size

                scaled_w = int((orig_w / orig_h) * target_height)

                if scaled_w > max_w:
                    max_w = scaled_w
        except Exception as e:
            print(f"Fehler bei {img_path.name}: {e}")

    final_max_w = math.ceil(max_w / 32) * 32

    print("-" * 30)
    print(f"Maximale skalierte Breite: {max_w} px")
    print(f"Empfohlene Breite für Swin (32er Stride): {final_max_w} px")
    print("-" * 30)

    return final_max_w


if __name__ == "__main__":
    import math

    ORDNER = "/tmp/unsloth/exp1_page/images"
    GEWUENSCHTE_HOEHE = 800

    max_width = get_max_scaled_width(ORDNER, GEWUENSCHTE_HOEHE)
