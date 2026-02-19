import shutil
import json
import uuid
from pathlib import Path
from PIL import Image
import shutil
import json
import base64
from pathlib import Path
from PIL import Image
import database.file_formats.pcgts as ns_pcgts
from database.file_formats.pcgts import Rect, MusicSymbol
from omr.end2end.codec.data_generator import process_single_block


def export_to_unsloth_dataset(pcgts_list, export_path, melody=False, mode="page"):
    base_dir = Path(export_path)
    img_dir = base_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jsonl_entries = []
    filename_mapping = {}

    for p in pcgts_list:
        page = p.page
        ds_page = p.dataset_page()
        orig_path = Path(ds_page.file("color_original").local_path())

        music_blocks = sorted(
            [b for b in page.blocks_of_type([ns_pcgts.BlockType.MUSIC])],
            key=lambda b: b.aabb.top()
        )

        if mode == "page":

            page_transcriptions = []
            for block in music_blocks:
                connections = [c for c in page.annotations.connections if c.music_region == block]
                trans = process_single_block(page, block, connections, 0, 0, melody=melody)
                if trans:
                    page_transcriptions.append(f"[NewLine]{trans}")

            if not page_transcriptions: continue

            final_text = "\n".join(page_transcriptions)

            u_id = str(uuid.uuid4())
            target_filename = f"{u_id}.jpg"
            shutil.copy(orig_path, img_dir / target_filename)

            filename_mapping[u_id] = f"page_{page.p_id}_full"
            jsonl_entries.append(create_entry(target_filename, final_text, "full music page"))

        else:

            try:
                full_image = Image.open(orig_path)
            except:
                continue

            for i, block in enumerate(music_blocks):
                connections = [c for c in page.annotations.connections if c.music_region == block]
                if not connections: continue

                connected_text_regions = list(set(c.text_region for c in connections if c.text_region))
                if len(connected_text_regions) > 1:
                    connected_text_regions = [
                        min(connected_text_regions, key=lambda c: abs(c.aabb.top() - block.aabb.bottom()))]

                if not connected_text_regions: continue

                combined_aabb = block.aabb.union(connected_text_regions[0].aabb)
                left_pad, top_pad = 50, (1 / 8 * connected_text_regions[0].aabb.height())

                trans = process_single_block(page, block, connections, left_pad, top_pad, melody=melody)
                if not trans: continue

                crop_box = (
                    page.page_to_image_scale(combined_aabb.left()) - left_pad,
                    page.page_to_image_scale(combined_aabb.top() - top_pad),
                    page.page_to_image_scale(combined_aabb.right()) + left_pad,
                    page.page_to_image_scale(combined_aabb.bottom())
                )

                u_id = str(uuid.uuid4())
                target_filename = f"{u_id}.jpg"
                full_image.crop(crop_box).save(img_dir / target_filename)

                filename_mapping[u_id] = f"page_{page.p_id}_block_{i}"
                jsonl_entries.append(create_entry(target_filename, trans, "music and text from this image"))

    save_jsonl(base_dir / "train_data.jsonl", jsonl_entries)
    with open(base_dir / "filename_mapping.json", "w", encoding="utf-8") as f:
        json.dump(filename_mapping, f, indent=4, ensure_ascii=False)

def create_entry(filename, text, task_desc):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"images/{filename}"},
                    {"type": "text", "text": f"Transcribe the {task_desc}."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            }
        ]
    }


def save_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from database import DatabaseBook

    c = DatabaseBook("mul_2_rsync_gt2")
    d = DatabaseBook("Koeln_Dombibl_1001b_part_gt")
    e = DatabaseBook("Pa_14819_gt")
    f = DatabaseBook("Geesebook2_andreas1")

    b = DatabaseBook('Geesebook1_complete_fixed_ro')
    g = DatabaseBook("Graduel_Part_1_gt")
    h = DatabaseBook("Graduel_Part_2_gt")
    i = DatabaseBook("Graduel_Part_3_gt")
    pcgts = [ns_pcgts.PcGts.from_file(x.file('pcgts')) for y in [c, d, e, f, b] for x in y.pages() if
             x.page_progress().verified_allowed()]

    export_to_unsloth_dataset(pcgts, "/tmp/unsloth/exp1_page/", melody=False)
