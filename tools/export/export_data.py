from dataclasses import dataclass

from pandas.core.computation.expressions import set_test_mode

import database.file_formats.pcgts as ns_pcgts
from typing import List, NamedTuple, Union, Optional
import json
import uuid
from enum import Enum
import numpy as np

from database.file_formats.book.document import Document
from database.file_formats.pcgts import Page, Line, SymbolType, MusicSymbol
from database.file_formats.pcgts.page import SyllableConnector, Connection


def export_for_finetuning_v8(pcgts_list: List[ns_pcgts.PcGts]):
    output_lines = []
    line_counter = 1

    for p in pcgts_list:
        page = p.page
        music_blocks = [b for b in page.blocks_of_type([ns_pcgts.BlockType.MUSIC])]


        for block in music_blocks:
            connections = [c for c in page.annotations.connections if c.music_region == block]

            if not connections:
                continue

            symbols = []
            for line in block.lines:
                symbols += line.symbols

            all_sc = sum([c.syllable_connections for c in connections], [])
            all_sc.sort(key=lambda sc: sc.note.coord.x)

            line_elements = []
            current_sym_idx = 0
            chant_marker_placed = False

            first_anchor_idx = symbols.index(all_sc[0].note) if all_sc else len(symbols)
            if first_anchor_idx > 0:
                line_elements.append(f"*[{format_syms(symbols[0:first_anchor_idx])}]")

            for i, sc in enumerate(all_sc):
                is_document_start = False
                parent_conn = next((c for c in connections if sc in c.syllable_connections), None)
                if parent_conn and parent_conn.text_region:
                    if any(getattr(tl, 'document_start', False) for tl in parent_conn.text_region.lines):
                        is_document_start = True

                prefix = ""
                if is_document_start and not chant_marker_placed:
                    prefix = "[NEW_CHANT] "
                    chant_marker_placed = True

                start_pos = symbols.index(sc.note)
                if i + 1 < len(all_sc):
                    end_pos = symbols.index(all_sc[i + 1].note)
                else:
                    end_pos = len(symbols)

                syllable_symbols = symbols[start_pos:end_pos]
                sym_block = format_syms(syllable_symbols)

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
                output_lines.append(f"<line>{line_counter}. {content}</line>")
                line_counter += 1

    return "\n".join(output_lines)


import json
import os
import shutil
from pathlib import Path


def format_syms(symbol_list):
    """Encodes music symbols into the LLM-optimized (Type|Pos|GC) format."""
    res = []

    for s in symbol_list:
        s: MusicSymbol
        p_val = s.position_in_staff if hasattr(s, 'position_in_staff') else "0"
        if s.symbol_type == ns_pcgts.SymbolType.CLEF:
            t_val, g_val = f"C_{s.clef_type.value}", "N"
        elif s.symbol_type == ns_pcgts.SymbolType.ACCID:
            t_val, g_val = f"A_{s.accid_type.value}", "N"
        elif s.symbol_type == ns_pcgts.SymbolType.NOTE:
            t_val = f"N_{s.note_type.value}"
            g_val = str(s.graphical_connection.value) if s.graphical_connection is not None else "NONE"
        else:
            t_val, g_val = "UNK", "N"
        res.append(f"({t_val}|{p_val}|{g_val})")
    return "".join(res)


def process_single_page(p):
    """Generates the transcription string for one page using V8 logic."""
    output_lines = []
    line_counter = 1
    page = p.page
    music_blocks = [b for b in page.blocks_of_type([ns_pcgts.BlockType.MUSIC])]

    for block in music_blocks:
        text_region = page.closest_below_text_line_to_music_line(block.lines[0], only_lyric=True)
        aabb = block.aabb.union(text_region.aabb)
        connections = [c for c in page.annotations.connections if c.music_region == block]
        if not connections: continue

        symbols = [s for line in block.lines for s in line.symbols]
        all_sc = sum([c.syllable_connections for c in connections], [])
        all_sc.sort(key=lambda sc: sc.note.coord.x)
        if not all_sc: continue

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
            end_pos = symbols.index(all_sc[i + 1].note) if i + 1 < len(all_sc) else len(symbols)

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
            output_lines.append(f"<line>{line_counter}. {content}</line>")
            line_counter += 1

    return "\n".join(output_lines)


def export_to_unsloth_dataset(pcgts_list, export_path):
    base_dir = Path(export_path)
    img_dir = base_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jsonl_entries = []

    for p in pcgts_list:
        ds_page = p.dataset_page()
        orig_path = Path(ds_page.file("color_original").local_path())

        target_img_name = f"{p.page.p_id}_{orig_path.name}"
        shutil.copy(orig_path, img_dir / target_img_name)

        transcription = process_single_page(p)

        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"images/{target_img_name}"},
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

    with open(base_dir / "train_data.jsonl", "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dataset ready at {export_path}")











if __name__ == "__main__":
    from database import DatabaseBook

    b = DatabaseBook('Geesebook1_complete_fixed_ro')
    #books = [DatabaseBook("Graduel_Part_1_gt"), DatabaseBook("Graduel_Part_2_gt"), DatabaseBook("Graduel_Part_3_gt"),]
    #books = [DatabaseBook("Geesebook1_complete_fixed_ro")]
    c = DatabaseBook("mul_2_rsync_gt2")
    d = DatabaseBook("Koeln_Dombibl_1001b_part_gt")
    e = DatabaseBook("Pa_14819_gt")
    f = DatabaseBook("Geesebook2_andreas1")

    pcgts = [ns_pcgts.PcGts.from_file(x.file('pcgts'))  for y in [b,c,d,e,f] for x in y.pages()  if x.page_progress().verified_allowed()]
    export_to_unsloth_dataset(pcgts, "/tmp/unsloth/exp2/")
    #data = export_for_finetuning_v8(pcgts[:4])
    #print(data)
