#!/usr/bin/env python3
"""
Visualise enriched monodi position data on the original manuscript images.

For each folio the corresponding page image is loaded and annotated with:

  ●  Green  dot — monodi note directly matched to a PCGTS symbol
  ●  Orange dot — monodi note whose position was interpolated
  ●  Red    dot — PCGTS symbol that exists but was NOT matched to any monodi note
  ─  Thin blue rectangle  — bounding box of the PCGTS music line
  ─  Thin grey rectangle  — bounding box of the PCGTS music region (block)
  A⁴  Pitch label (base + octave) next to each dot

One output image is produced per page, regardless of how many chants
contribute annotations to it.

Usage
-----
With a DatabaseBook (preferred):

    python -m omr.steps.tools.monodi_pattern_search.draw_enrichment \\
        --monodi-path  /path/to/Pa14819_enriched \\
        --book         Pa_14819_gt \\
        --output-dir   /path/to/debug_images \\
        --folio-zfill  5 \\
        --folio-recto-suffix

With a raw image + PCGTS directory:

    python -m omr.steps.tools.monodi_pattern_search.draw_enrichment \\
        --monodi-path  /path/to/Pa14819_enriched \\
        --images-dir   /path/to/images \\
        --pcgts-dir    /path/to/book \\
        --output-dir   /path/to/debug_images \\
        --folio-zfill  5 \\
        --folio-recto-suffix

``--images-dir`` expects files named ``<page_name>.<ext>`` (jpg/png/tif).
``--pcgts-dir`` expects ``<pcgts-dir>/<page_name>/pcgts.json`` (same layout
as the ommr4all storage).  Omit ``--pcgts-dir`` to skip the red
unmatched-PCGTS layer.

Optional rendering flags
------------------------
--image-key KEY
    Image file to load from DatabaseBook (default: color_original).
    Alternatives: color_highres_preproc, color_lowres_preproc, color_norm.
--marker-radius N
    Fixed marker radius in pixels (default: auto = 1 %% of image height).
--no-line-boxes
    Do not draw music-line bounding rectangles.
--no-region-boxes
    Do not draw music-region bounding rectangles.
--no-labels
    Do not draw pitch labels next to markers.
--only-matched
    Draw only directly-matched monodi notes (skip interpolated and unmatched).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Make the server module importable when run directly
# ---------------------------------------------------------------------------
_SERVER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from database.file_formats.importer.mondodi.monodi_parser import (
    load_manuscript,
    parse_data_json_by_page,
)

# ---------------------------------------------------------------------------
# Colours  (RGB tuples; alpha added inline as needed)
# ---------------------------------------------------------------------------
COLOR_MATCHED        = (0,   200,  50)    # green  — monodi note, direct match
COLOR_INTERPOLATED   = (255, 140,   0)    # orange — monodi note, interpolated
COLOR_UNMATCHED_PCGTS = (220,  30,  30)  # red    — PCGTS note without a monodi match
COLOR_LINE_BOX       = (50,  130, 220)    # blue   — music-line bbox
COLOR_REGION_BOX     = (140, 140, 140)   # grey   — music-region bbox
COLOR_LABEL          = (20,   20,  20)    # near-black text

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoteAnnotation:
    """Rendering data for one enriched monodi note."""
    base:             str
    octave:           int
    note_type:        str
    x:                float    # height-normalised
    y:                float    # height-normalised
    matched:          bool
    line_coords:      str      # height-normalised polygon string
    region_coords:    str
    pcgts_symbol_id:  str = "" # set for direct matches; empty for interpolated


@dataclass
class UnmatchedPcgtsNote:
    """A PCGTS note that was not matched to any monodi note."""
    x:         float    # height-normalised
    y:         float    # height-normalised
    note_name: str      # "A"–"G" or "?"


@dataclass
class LineUnmatchedIndicator:
    """
    Count of monodi notes on one PCGTS music line that received no position.
    Drawn as a badge on the left/right margin next to the music line.
    """
    line_coords: str   # height-normalised polygon — used to locate the line
    count: int         # number of unpositioned monodi notes on this line


# ---------------------------------------------------------------------------
# JSON walking helpers
# ---------------------------------------------------------------------------

def _collect_positioned_notes(data_json: dict) -> Dict[str, NoteAnnotation]:
    """Return ``{uuid: NoteAnnotation}`` for every note carrying a position."""
    result: Dict[str, NoteAnnotation] = {}

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("kind") == "Syllable":
            for spaced in node.get("notes", {}).get("spaced", []):
                for ns in spaced.get("nonSpaced", []):
                    for grouped in ns.get("grouped", []):
                        uuid = grouped.get("uuid", "")
                        pos  = grouped.get("position")
                        if uuid and pos:
                            result[uuid] = NoteAnnotation(
                                base=grouped.get("base", "?"),
                                octave=grouped.get("octave", 4),
                                note_type=grouped.get("noteType", "Normal"),
                                x=float(pos.get("x", 0)),
                                y=float(pos.get("y", 0)),
                                matched=bool(pos.get("matched", False)),
                                line_coords=pos.get("line_coords", ""),
                                region_coords=pos.get("region_coords", ""),
                                pcgts_symbol_id=pos.get("pcgts_symbol_id", ""),
                            )
        for child in node.get("children", []):
            _walk(child)

    _walk(data_json)
    return result


# ---------------------------------------------------------------------------
# PCGTS loading helpers
# ---------------------------------------------------------------------------

_IDX_TO_LETTER = ["A", "B", "C", "D", "E", "F", "G"]  # NoteName: A=0…G=6


def _pcgts_notes_from_json(page_json: dict) -> List[Tuple[str, float, float, str]]:
    """
    Extract ``(symbol_id, x, y, note_name)`` for every note in a raw PCGTS
    page JSON dict (height-normalised coordinates).
    """
    notes = []
    for block in page_json.get("blocks", []):
        if block.get("type") != "music":
            continue
        for line in block.get("lines", []):
            for sym in line.get("symbols", []):
                if sym.get("type") != "note":
                    continue
                coord = sym.get("coord", "0,0").split(",")
                try:
                    x = float(coord[0])
                    y = float(coord[1])
                except (IndexError, ValueError):
                    x = y = 0.0
                try:
                    name = _IDX_TO_LETTER[int(sym["pname"])]
                except (KeyError, IndexError, ValueError, TypeError):
                    name = "?"
                notes.append((sym.get("id", ""), x, y, name))
    return notes


def _pcgts_notes_from_db_page(db_page) -> List[Tuple[str, float, float, str]]:
    """
    Extract ``(symbol_id, x, y, note_name)`` using the typed DatabasePage API.
    Calls ``page.update_note_names()`` via ``from_json`` / constructor.
    """
    from database.file_formats.pcgts.page.musicsymbol import SymbolType, NoteName
    from database.file_formats.pcgts.page.definitions import BlockType

    page = db_page.pcgts().page
    notes = []
    for block in page.blocks:
        if block.block_type != BlockType.MUSIC:
            continue
        for line in block.lines:
            for sym in line.symbols:
                if sym.symbol_type != SymbolType.NOTE:
                    continue
                if sym.note_name == NoteName.UNDEFINED:
                    continue
                x = sym.coord.x if sym.coord else 0.0
                y = sym.coord.y if sym.coord else 0.0
                notes.append((sym.id, x, y, sym.note_name.name))
    return notes


def _music_line_coords_from_json(page_json: dict) -> List[str]:
    """
    Return a top-to-bottom sorted list of coord strings for every music line
    in a raw PCGTS page JSON dict.  Index 0 = topmost line on the page.
    """
    lines: List[dict] = []
    for block in page_json.get("blocks", []):
        if block.get("type") != "music":
            continue
        for line in block.get("lines", []):
            lines.append(line)

    def _avg_y(line: dict) -> float:
        ys = []
        for pt in line.get("coords", "").strip().split():
            parts = pt.split(",")
            if len(parts) == 2:
                try:
                    ys.append(float(parts[1]))
                except ValueError:
                    pass
        return sum(ys) / len(ys) if ys else 0.0

    lines.sort(key=_avg_y)
    return [line.get("coords", "") for line in lines]


def _music_line_coords_from_db_page(db_page) -> List[str]:
    """
    Return a top-to-bottom sorted list of coord strings for every music line
    using the typed DatabasePage API.
    """
    from database.file_formats.pcgts.page.definitions import BlockType

    page = db_page.pcgts().page
    entries: List[Tuple[float, str]] = []
    for block in page.blocks:
        if block.block_type != BlockType.MUSIC:
            continue
        for line in block.lines:
            coords_str = line.coords.to_string() if line.coords is not None else ""
            ys = []
            for pt in coords_str.strip().split():
                parts = pt.split(",")
                if len(parts) == 2:
                    try:
                        ys.append(float(parts[1]))
                    except ValueError:
                        pass
            avg_y = sum(ys) / len(ys) if ys else 0.0
            entries.append((avg_y, coords_str))
    entries.sort(key=lambda e: e[0])
    return [e[1] for e in entries]


def _load_pcgts_json(pcgts_dir: str, page_name: str) -> Optional[dict]:
    """Load a raw pcgts.json from a directory, trying with and without leading zeros."""
    path = os.path.join(pcgts_dir, page_name, "pcgts.json")
    if not os.path.exists(path):
        plain = page_name.lstrip("0") or "0"
        path = os.path.join(pcgts_dir, plain, "pcgts.json")
        if not os.path.exists(path):
            return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _parse_coords_str(coords_str: str) -> List[Tuple[float, float]]:
    pts = []
    for token in coords_str.strip().split():
        parts = token.split(",")
        if len(parts) == 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def _coords_bbox(coords_str: str, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    pts = _parse_coords_str(coords_str)
    if not pts:
        return None
    xs = [p[0] * img_h for p in pts]
    ys = [p[1] * img_h for p in pts]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


# ---------------------------------------------------------------------------
# Folio resolution  (mirrors run_enrichment.py)
# ---------------------------------------------------------------------------

def _resolve_folio(
    folio: str,
    folio_offset: int,
    folio_map: Optional[Dict[str, str]],
    folio_zfill: int,
    folio_recto_suffix: bool,
) -> Optional[str]:
    if folio_map is not None:
        return folio_map.get(str(folio), None)
    folio_clean = re.sub(r'^f(?:ol)?\.?\s*', '', folio.strip(), flags=re.IGNORECASE)
    suffix = ""
    if folio_recto_suffix:
        suffix = "v" if folio_clean.endswith(("v", "V")) else "r"
    folio_num_str = folio_clean.rstrip("vrVR")
    try:
        page_idx = int(folio_num_str) + folio_offset
        return str(page_idx).zfill(folio_zfill) + suffix
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_IMAGE_FILE_IDS   = ["color_original", "color_highres_preproc",
                     "color_lowres_preproc", "color_norm"]


def _image_path_from_book(db_page, image_key: str) -> Optional[str]:
    keys_to_try = [image_key] + [k for k in _IMAGE_FILE_IDS if k != image_key]
    for key in keys_to_try:
        try:
            path = db_page.file(key).local_path()
            if os.path.exists(path):
                return path
        except Exception:
            pass
    return None


def _image_path_from_dir(images_dir: str, page_name: str) -> Optional[str]:
    for ext in _IMAGE_EXTENSIONS:
        path = os.path.join(images_dir, page_name + ext)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

COLOR_INDICATOR_BG   = (160,  30, 160)   # purple badge background
COLOR_INDICATOR_TEXT = (255, 255, 255)   # white badge text


def _draw_line_indicators(
    draw:       ImageDraw.ImageDraw,
    indicators: List[LineUnmatchedIndicator],
    img_w:      int,
    img_h:      int,
    font,
    radius:     int,
) -> None:
    """
    Draw a small badge showing the number of unpositioned monodi notes next to
    each music line.  The badge is placed on the left side of the line if there
    is enough room, otherwise on the right.
    """
    pad      = max(3, radius // 2)
    badge_h  = max(14, radius * 2)

    for ind in indicators:
        if ind.count <= 0 or not ind.line_coords:
            continue
        bbox = _coords_bbox(ind.line_coords, img_h)
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        center_y = (top + bottom) // 2

        text = str(ind.count)
        # Measure text dimensions
        try:
            bb = font.getbbox(text)      # (left, top, right, bottom) in font space
            text_w = bb[2] - bb[0]
            text_h = bb[3] - bb[1]
            font_offset_x, font_offset_y = bb[0], bb[1]
        except AttributeError:
            text_w = len(text) * (badge_h // 2)
            text_h = badge_h
            font_offset_x = font_offset_y = 0

        badge_w = text_w + pad * 2

        # Prefer left side; fall back to right when line touches the image edge
        if left >= badge_w + pad:
            bx0 = left - badge_w - pad
        else:
            bx0 = right + pad
        bx1 = bx0 + badge_w
        by0 = center_y - badge_h // 2
        by1 = by0 + badge_h

        # Badge background with rounded corners
        r = badge_h // 3
        draw.rounded_rectangle(
            [(bx0, by0), (bx1, by1)],
            radius=r,
            fill=COLOR_INDICATOR_BG + (220,),
        )

        # Thin connecting line from badge to music-line left edge
        draw.line(
            [(bx1, center_y), (left, center_y)],
            fill=COLOR_INDICATOR_BG + (160,),
            width=max(1, radius // 5),
        )

        # Text centred inside badge
        tx = bx0 + pad - font_offset_x
        ty = center_y - text_h // 2 - font_offset_y
        draw.text((tx, ty), text, fill=COLOR_INDICATOR_TEXT + (255,), font=font)


def _draw_annotations(
    image_path: str,
    annotations: List[NoteAnnotation],
    unmatched_pcgts: List[UnmatchedPcgtsNote],
    indicators: List[LineUnmatchedIndicator],
    output_path: str,
    *,
    marker_radius: Optional[int],
    draw_line_boxes:   bool,
    draw_region_boxes: bool,
    draw_labels:       bool,
    only_matched:      bool,
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    radius = marker_radius if marker_radius is not None else max(4, img_h // 100)
    label_offset = radius + 3

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            size=max(10, radius * 2),
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    # --- Bounding boxes (drawn behind dots) ---
    drawn_line_boxes:   Set[str] = set()
    drawn_region_boxes: Set[str] = set()

    for ann in annotations:
        if only_matched and not ann.matched:
            continue
        if draw_region_boxes and ann.region_coords and ann.region_coords not in drawn_region_boxes:
            bbox = _coords_bbox(ann.region_coords, img_h)
            if bbox:
                draw.rectangle(bbox, outline=COLOR_REGION_BOX + (160,), width=2)
            drawn_region_boxes.add(ann.region_coords)
        if draw_line_boxes and ann.line_coords and ann.line_coords not in drawn_line_boxes:
            bbox = _coords_bbox(ann.line_coords, img_h)
            if bbox:
                draw.rectangle(bbox, outline=COLOR_LINE_BOX + (200,), width=2)
            drawn_line_boxes.add(ann.line_coords)

    # --- Unmatched PCGTS notes: red hollow circle with cross ---
    if not only_matched:
        for note in unmatched_pcgts:
            px = int(note.x * img_h)
            py = int(note.y * img_h)
            r  = max(3, radius - 1)
            draw.ellipse(
                [(px - r, py - r), (px + r, py + r)],
                fill=None,
                outline=COLOR_UNMATCHED_PCGTS + (230,),
                width=max(2, r // 3),
            )
            # small cross inside
            arm = max(2, r // 2)
            draw.line([(px - arm, py), (px + arm, py)],
                      fill=COLOR_UNMATCHED_PCGTS + (230,), width=max(1, r // 4))
            draw.line([(px, py - arm), (px, py + arm)],
                      fill=COLOR_UNMATCHED_PCGTS + (230,), width=max(1, r // 4))
            if draw_labels and note.note_name != "?":
                draw.text(
                    (px + label_offset, py - r),
                    note.note_name,
                    fill=COLOR_UNMATCHED_PCGTS + (200,),
                    font=font,
                )

    # --- Matched / interpolated monodi notes: filled circles ---
    for ann in annotations:
        if only_matched and not ann.matched:
            continue
        px = int(ann.x * img_h)
        py = int(ann.y * img_h)
        color = COLOR_MATCHED if ann.matched else COLOR_INTERPOLATED
        draw.ellipse(
            [(px - radius, py - radius), (px + radius, py + radius)],
            fill=color + (210,),
            outline=color + (255,),
            width=max(1, radius // 4),
        )
        if draw_labels:
            label = f"{ann.base}{ann.octave}"
            draw.text(
                (px + label_offset, py - radius),
                label,
                fill=COLOR_LABEL + (220,),
                font=font,
            )

    # --- Line unmatched-count badges (drawn on top of everything) ---
    _draw_line_indicators(draw, indicators, img_w, img_h, font, radius)

    img.save(output_path)


# ---------------------------------------------------------------------------
# Django / DatabaseBook helpers
# ---------------------------------------------------------------------------

def _setup_django() -> None:
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        return
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ommr4all.settings")
    import django
    django.setup()


def _build_page_index(book_name: str) -> Dict[str, Any]:
    from database.database_book import DatabaseBook
    return {p.page: p for p in DatabaseBook(book_name).pages()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw enriched monodi positions onto manuscript page images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--monodi-path", required=True,
        help="Path to the ENRICHED manuscript folder (output of run_enrichment.py).",
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--book",
        help="DatabaseBook name — images and PCGTS are loaded from ommr4all storage.",
    )
    src_group.add_argument(
        "--images-dir",
        help="Directory of page images named <page_name>.<ext> (jpg/png/tif).",
    )

    parser.add_argument(
        "--pcgts-dir", default=None,
        help="(--images-dir mode only) Directory with <page_name>/pcgts.json files. "
             "Required to draw unmatched PCGTS notes in red. "
             "Omit to skip that layer.",
    )
    parser.add_argument("--output-dir", required=True,
                        help="Destination directory for annotated images.")

    # Folio resolution
    parser.add_argument("--folio-offset",      type=int, default=0)
    parser.add_argument("--folio-zfill",        type=int, default=4, metavar="N")
    parser.add_argument("--folio-recto-suffix", action="store_true")
    parser.add_argument(
        "--folio-map", default=None,
        help='JSON object mapping folio IDs to page names.',
    )

    # Rendering
    parser.add_argument(
        "--image-key", default="color_original", metavar="KEY",
        help="Image file ID to load from DatabaseBook (default: color_original).",
    )
    parser.add_argument(
        "--marker-radius", type=int, default=None, metavar="N",
        help="Dot radius in pixels (default: auto = 1%% of image height).",
    )
    parser.add_argument("--no-line-boxes",   action="store_true")
    parser.add_argument("--no-region-boxes", action="store_true")
    parser.add_argument("--no-labels",       action="store_true")
    parser.add_argument("--only-matched",    action="store_true",
                        help="Draw only directly-matched monodi notes.")

    args = parser.parse_args()

    folio_map: Optional[Dict[str, str]] = None
    if args.folio_map:
        try:
            folio_map = json.loads(args.folio_map)
        except json.JSONDecodeError as exc:
            parser.error(f"--folio-map is not valid JSON: {exc}")

    # Set up page / PCGTS index
    page_index: Optional[Dict[str, Any]] = None
    if args.book:
        print(f"Initialising Django and loading book '{args.book}' …")
        _setup_django()
        page_index = _build_page_index(args.book)
        print(f"  {len(page_index)} pages found.")
    else:
        if not os.path.isdir(args.images_dir):
            parser.error(f"--images-dir does not exist: {args.images_dir}")
        if args.pcgts_dir and not os.path.isdir(args.pcgts_dir):
            parser.error(f"--pcgts-dir does not exist: {args.pcgts_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Pass 1: collect monodi annotations per folio
    # -----------------------------------------------------------------------
    # folio → list of NoteAnnotation
    folio_annotations:    Dict[str, List[NoteAnnotation]]   = defaultdict(list)
    # folio → set of PCGTS symbol IDs that were directly matched
    folio_matched_ids:    Dict[str, Set[str]]               = defaultdict(set)
    # folio → {line_number: unpositioned_count}  (aggregated across chants)
    # line_number is 1-based (same as MonodiPageLine.line_number).
    # Stored as int so Pass 2 can index PCGTS lines with line_number - 1.
    folio_line_unpos:     Dict[str, Dict[int, int]]         = defaultdict(lambda: defaultdict(int))
    # folio → resolved page name
    folio_to_page:        Dict[str, str]                    = {}

    print(f"\nScanning enriched manuscript: {args.monodi_path}")
    manuscript = load_manuscript(args.monodi_path)
    print(f"  '{manuscript.meta.id}' — {len(manuscript.chants)} chants")

    for chant in manuscript.chants:
        chant_folder = os.path.join(args.monodi_path, chant.meta.id)
        data_path    = os.path.join(chant_folder, "data.json")
        if not os.path.exists(data_path):
            continue

        with open(data_path, "r", encoding="utf-8") as fh:
            data_json = json.load(fh)

        positioned = _collect_positioned_notes(data_json)
        if not positioned:
            continue

        try:
            zeilenstart = int(chant.meta.zeilenstart)
        except (ValueError, TypeError):
            zeilenstart = 1

        page_lines = parse_data_json_by_page(
            data_json,
            foliostart=chant.meta.foliostart,
            zeilenstart=zeilenstart,
        )

        for pl in page_lines:
            folio = str(pl.folio)
            if folio not in folio_to_page:
                page_name = _resolve_folio(
                    folio, args.folio_offset, folio_map,
                    args.folio_zfill, args.folio_recto_suffix,
                )
                if page_name:
                    folio_to_page[folio] = page_name

            # Collect per-line unpositioned count for this page_line
            line_anns: List[NoteAnnotation] = []
            unpos_count = 0
            for note in pl.all_notes:
                if note.uuid in positioned:
                    ann = positioned[note.uuid]
                    folio_annotations[folio].append(ann)
                    if ann.pcgts_symbol_id:
                        folio_matched_ids[folio].add(ann.pcgts_symbol_id)
                    line_anns.append(ann)
                else:
                    unpos_count += 1

            if unpos_count > 0:
                # Key by line_number (1-based) so Pass 2 can look up the
                # PCGTS line geometry even when no notes were positioned.
                folio_line_unpos[folio][pl.line_number] += unpos_count

    # -----------------------------------------------------------------------
    # Pass 2: per folio — load PCGTS, find unmatched notes, draw
    # -----------------------------------------------------------------------
    total_drawn   = 0
    total_skipped = 0

    folios_sorted = sorted(folio_annotations.keys(),
                           key=lambda f: folio_to_page.get(f, f))

    print(f"\nDrawing annotations for {len(folios_sorted)} unique folios …\n")

    for folio in folios_sorted:
        annotations  = folio_annotations[folio]
        matched_ids  = folio_matched_ids[folio]
        page_name    = folio_to_page.get(folio)

        if page_name is None:
            print(f"  ⚠  Folio '{folio}' — unresolvable; skipped")
            total_skipped += 1
            continue

        # Locate image
        image_path: Optional[str] = None
        db_page = None
        if page_index is not None:
            db_page    = page_index.get(page_name)
            if db_page is None:
                print(f"  ⚠  Folio '{folio}' → '{page_name}' — not in book; skipped")
                total_skipped += 1
                continue
            image_path = _image_path_from_book(db_page, args.image_key)
        else:
            image_path = _image_path_from_dir(args.images_dir, page_name)

        if not image_path:
            print(f"  ⚠  Folio '{folio}' → '{page_name}' — image not found; skipped")
            total_skipped += 1
            continue

        # Load raw PCGTS JSON once (used for both unmatched-note layer and
        # line-indicator geometry when not in DatabaseBook mode).
        raw_pcgts: Optional[dict] = None
        raw_page_node: Optional[dict] = None
        if page_index is None and args.pcgts_dir:
            raw_pcgts = _load_pcgts_json(args.pcgts_dir, page_name)
            if raw_pcgts is not None:
                raw_page_node = raw_pcgts.get("page", raw_pcgts)

        # Collect unmatched PCGTS notes
        unmatched_pcgts: List[UnmatchedPcgtsNote] = []
        if not args.only_matched:
            all_pcgts_notes: List[Tuple[str, float, float, str]] = []
            if db_page is not None:
                try:
                    all_pcgts_notes = _pcgts_notes_from_db_page(db_page)
                except Exception as exc:
                    print(f"    ⚠  Could not load PCGTS for '{page_name}': {exc}")
            elif raw_page_node is not None:
                all_pcgts_notes = _pcgts_notes_from_json(raw_page_node)

            for sym_id, x, y, note_name in all_pcgts_notes:
                if sym_id not in matched_ids:
                    unmatched_pcgts.append(UnmatchedPcgtsNote(x=x, y=y, note_name=note_name))

        ext      = os.path.splitext(image_path)[1] or ".jpg"
        out_name = f"{page_name}_annotated{ext}"
        out_path = os.path.join(args.output_dir, out_name)

        # Build per-line unmatched indicators for this folio.
        # Resolve each line_number (1-based) to its PCGTS music-line coords
        # by looking it up in the sorted list of lines from the PCGTS page.
        indicators: List[LineUnmatchedIndicator] = []
        unpos_by_line_number = folio_line_unpos.get(folio, {})
        if unpos_by_line_number:
            pcgts_line_coords_list: List[str] = []
            if db_page is not None:
                try:
                    pcgts_line_coords_list = _music_line_coords_from_db_page(db_page)
                except Exception as exc:
                    print(f"    ⚠  Could not load music-line coords for '{page_name}': {exc}")
            elif raw_page_node is not None:
                pcgts_line_coords_list = _music_line_coords_from_json(raw_page_node)

            for line_number, cnt in sorted(unpos_by_line_number.items()):
                if cnt <= 0:
                    continue
                line_idx = line_number - 1
                if 0 <= line_idx < len(pcgts_line_coords_list):
                    lc = pcgts_line_coords_list[line_idx]
                else:
                    lc = ""   # out of range — indicator drawn without geometry
                indicators.append(LineUnmatchedIndicator(line_coords=lc, count=cnt))
        n_unpos_total = sum(ind.count for ind in indicators)

        n_matched      = sum(1 for a in annotations if a.matched)
        n_interpolated = sum(1 for a in annotations if not a.matched)
        print(f"  {folio:<14} → {page_name}   "
              f"monodi: {len(annotations)} ({n_matched} matched, "
              f"{n_interpolated} interp., {n_unpos_total} unpositioned)   "
              f"pcgts unmatched: {len(unmatched_pcgts)}   "
              f"→ {out_name}")

        _draw_annotations(
            image_path, annotations, unmatched_pcgts, indicators, out_path,
            marker_radius=args.marker_radius,
            draw_line_boxes=not args.no_line_boxes,
            draw_region_boxes=not args.no_region_boxes,
            draw_labels=not args.no_labels,
            only_matched=args.only_matched,
        )
        total_drawn += 1

    print(f"\nDone.  {total_drawn} images written to: {args.output_dir}"
          + (f"   ({total_skipped} skipped)" if total_skipped else ""))


if __name__ == "__main__":
    main()
