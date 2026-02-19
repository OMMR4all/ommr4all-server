import uuid
import database.file_formats.pcgts as ns_pcgts
from tools.export.export_data import format_syms
import json
from pathlib import Path
from PIL import Image


def process_single_block(page, block, connections):
    """Generates the transcription string for a specific music block."""

    symbols = [s for line in block.lines for s in line.symbols]
    all_sc = sum([c.syllable_connections for c in connections], [])

    if not all_sc:
        return None


    try:
        all_sc.sort(key=lambda sc: symbols.index(sc.note))
    except ValueError:
        valid_sc = []
        for sc in all_sc:
            if sc.note in symbols:
                valid_sc.append(sc)
        all_sc = sorted(valid_sc, key=lambda sc: symbols.index(sc.note))

    if not all_sc:
        return None

    line_elements = []
    chant_marker_placed = False

    first_anchor_idx = symbols.index(all_sc[0].note)
    if first_anchor_idx > 0:
        line_elements.append(f"*[{format_syms(symbols[0:first_anchor_idx])}]")

    for i, sc in enumerate(all_sc):
        is_document_start = False

        parent_conn = next((c for c in connections if sc in c.syllable_connections), None)

        if parent_conn and parent_conn.text_region:
            if any(getattr(tl, 'document_start', False) for tl in parent_conn.text_region.lines):
                is_document_start = True

        prefix = "[NEW_CHANT] " if is_document_start and not chant_marker_placed else ""
        if prefix: chant_marker_placed = True

        start_pos = symbols.index(sc.note)

        if i + 1 < len(all_sc):
            end_pos = symbols.index(all_sc[i + 1].note)
        else:
            end_pos = len(symbols)

        sym_block = format_syms(symbols[start_pos:end_pos])
        syl_text = sc.syllable.text.replace("-", "").strip() or "_"
        formatted_chunk = f"{prefix}{syl_text}[{sym_block}]"

        if not line_elements:
            line_elements.append(formatted_chunk)
        elif sc.syllable.connection == sc.syllable.connection.NEW:
            line_elements.append(f" {formatted_chunk}")
        else:
            line_elements.append(f"-{formatted_chunk}")

    if line_elements:
        content = "".join(line_elements).strip()
        return f"<line>1. {content}</line>"

    return None


def export_to_unsloth_dataset(pcgts_list, export_path):
    base_dir = Path(export_path)
    img_dir = base_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jsonl_entries = []
    filename_mapping = {}

    print(f"Processing {len(pcgts_list)} pages...")

    for p in pcgts_list:
        page = p.page
        ds_page = p.dataset_page()
        orig_path = Path(ds_page.file("color_original").local_path())

        try:
            full_image = Image.open(orig_path)
        except Exception as e:
            print(f"Error opening image {orig_path}: {e}")
            continue

        music_blocks = [b for b in page.blocks_of_type([ns_pcgts.BlockType.MUSIC])]

        for i, block in enumerate(music_blocks):
            try:
                connections = [c for c in page.annotations.connections if c.music_region == block]
                if not connections:
                    continue

                connected_text_regions = list(set(c.text_region for c in connections if c.text_region))

                if not connected_text_regions:
                    continue

                combined_aabb = block.aabb
                for tr in connected_text_regions:
                    combined_aabb = combined_aabb.union(tr.aabb)

                transcription = process_single_block(page, block, connections)
                if not transcription:
                    continue

                crop_box = (
                    page.page_to_image_scale(combined_aabb.left())-50,
                    page.page_to_image_scale(combined_aabb.top()-(1/8* connected_text_regions[0].aabb.height())),
                    page.page_to_image_scale(combined_aabb.right())+50,
                    page.page_to_image_scale(combined_aabb.bottom())
                )
                cropped_img = full_image.crop(crop_box)
                uuid4 = uuid.uuid4()
                readable_id = f"local_{page.location.page}_pageid_{page.p_id}_block_{i}"

                target_filename = f"{uuid4}.jpg"

                filename_mapping[str(uuid4)] = readable_id

                save_path = img_dir / target_filename
                cropped_img.save(save_path)

                entry = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"images/{target_filename}"},
                                {"type": "text", "text": "Transcribe the music and text from this image."}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": transcription}
                            ]
                        }
                    ]
                }
                jsonl_entries.append(entry)

            except Exception as e:
                print(f"Error processing block {i} on page {page.p_id}: {e}")
                continue

    # Save outputs
    with open(base_dir / "train_data.jsonl", "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(base_dir / "filename_mapping.json", "w", encoding="utf-8") as f:
        json.dump(filename_mapping, f, indent=4, ensure_ascii=False)

    print(f"Dataset ready at {export_path}")
    print(f"Generated {len(jsonl_entries)} entries.")
if __name__ == "__main__":
    from database import DatabaseBook

    c = DatabaseBook("mul_2_rsync_gt2")
    d = DatabaseBook("Koeln_Dombibl_1001b_part_gt")
    e = DatabaseBook("Pa_14819_gt")
    f = DatabaseBook("Geesebook2_andreas1")
    pcgts = [ns_pcgts.PcGts.from_file(x.file('pcgts'))  for y in [c,d,e,f] for x in y.pages() if x.page_progress().verified_allowed()]
    export_to_unsloth_dataset(pcgts, "/tmp/unsloth/exp5_blocks/")