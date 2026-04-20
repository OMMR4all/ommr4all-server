#!/usr/bin/env python3
"""
Enrich monodi+ export files with position data from ommr4all PCGTS files.

Usage
-----
Run from the repo root (or any directory where the server module is importable):

**Using a DatabaseBook (preferred — resolves pitches from clef context):**

    python -m omr.steps.tools.monodi_pattern_search.run_enrichment \\
        --monodi-path  /path/to/Pa\\ 1213          \\
        --book         Koeln_Dombibl_1001b_part_gt \\
        --output-path  /path/to/Pa\\ 1213_enriched \\
        [--folio-offset 0]                          \\
        [--folio-map    '{"86": "0085", "86v": "0086"}'] \\
        [--include-note-types Normal Liquescent]

**Using a raw PCGTS directory (fallback):**

    python -m omr.steps.tools.monodi_pattern_search.run_enrichment \\
        --monodi-path  /path/to/Pa\\ 1213          \\
        --pcgts-dir    /path/to/ommr4all/book      \\
        --output-path  /path/to/Pa\\ 1213_enriched \\
        [--folio-offset 0]                          \\
        [--folio-map    '{"86": "0085", "86v": "0086"}'] \\
        [--include-note-types Normal Liquescent]

Arguments
---------
--monodi-path
    Path to the manuscript export folder (must contain meta.json and one
    sub-folder per chant, each with meta.json + data.json).

--book
    Name of the ommr4all DatabaseBook (directory name under the configured
    PRIVATE_MEDIA_ROOT storage).  When given, Django is initialised and pages
    are loaded via the ``DatabaseBook`` / ``DatabasePage`` API so that pitch
    names are resolved correctly from clef context.  Mutually exclusive with
    ``--pcgts-dir``.

--pcgts-dir
    Path to the ommr4all book directory (fallback when ``--book`` is not
    given).  The script expects PCGTS files at::

        <pcgts-dir>/<page-name>/pcgts.json

    where <page-name> is derived from the folio number (see --folio-offset /
    --folio-map below).

--output-path
    Destination folder.  The enriched data.json files are written there,
    preserving the original chant-folder structure.  The manuscript meta.json
    is copied unchanged.  The folder is created if it does not exist.

--folio-offset  (default: 0)
    Integer added to the folio number to obtain the zero-based PCGTS page
    index.  The page name is then formatted as a 4-digit zero-padded string::

        page_name = str(int(folio) + folio_offset).zfill(4)

    Example: folio "86", offset -1  →  page "0085".
    Ignored when --folio-map is provided.

--folio-map  (default: none)
    JSON object that maps folio identifiers to PCGTS page names, overriding
    --folio-offset completely.  Folios not listed in the map are skipped.
    Example::

        '{"18": "0017", "19": "0018", "19v": "0019"}'

--include-note-types  (default: all types included)
    Space-separated list of monodi noteType values to include in the
    alignment.  Notes with other types are not aligned and receive no
    position.  Example: ``Normal Liquescent Oriscus Strophicus``

Output format
-------------
Each enriched note node gains a ``"position"`` key::

    {
        "uuid": "...",
        "base": "G",
        "octave": 4,
        "noteType": "Liquescent",
        "liquescent": true,
        "focus": false,
        "position": {
            "x":                 0.1234,   # height-normalised
            "y":                 0.4567,   # height-normalised
            "position_in_staff": 5,        # int (see MusicSymbolPositionInStaff)
            "line_id":           "<uuid>", # PCGTS music-line id
            "line_coords":       "...",    # height-norm. polygon of the staff line
            "region_id":         "<uuid>", # PCGTS music-block (region) id
            "region_coords":     "...",    # height-norm. polygon of the music region
            "matched":           true      # false = interpolated between neighbours
        }
    }

Notes that could not be matched or interpolated receive no ``"position"`` key.

A summary is printed to stdout after processing each chant.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Optional

# Make the server module importable when run directly
_SERVER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from database.file_formats.importer.mondodi.monodi_parser import (
    MonodiChant,
    load_manuscript,
)
from omr.steps.tools.monodi_pattern_search.position_enricher import (
    MonodiPositionEnricher,
    alignment_stats,
)


# ---------------------------------------------------------------------------
# Django / DatabaseBook setup  (only when --book is used)
# ---------------------------------------------------------------------------

def _setup_django() -> None:
    """Initialise Django settings so DatabaseBook/DatabasePage work."""
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        return  # already configured by the caller
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ommr4all.settings")
    import django
    django.setup()


def _build_page_index(book_name: str) -> Dict[str, Any]:
    """
    Return a ``{page_name: DatabasePage}`` dict for every page in the book.

    Page names are the directory names returned by ``DatabaseBook.pages()``,
    e.g. ``"0001"``, ``"0002"``, etc.
    """
    from database.database_book import DatabaseBook
    book = DatabaseBook(book_name)
    return {p.page: p for p in book.pages()}


# ---------------------------------------------------------------------------
# Folio → PCGTS page name resolution
# ---------------------------------------------------------------------------

def _resolve_folio(
    folio: str,
    folio_offset: int,
    folio_map: Optional[Dict[str, str]],
    folio_zfill: int = 4,
    folio_recto_suffix: bool = False,
) -> Optional[str]:
    """
    Convert a folio identifier to a PCGTS page name.

    Parameters
    ----------
    folio:
        Folio string from monodi (e.g. ``"126"``, ``"126v"``).
    folio_offset:
        Integer added to the numeric part of the folio before formatting.
    folio_map:
        When provided, used as a direct lookup table; all other parameters
        are ignored.
    folio_zfill:
        Width to which the numeric part is zero-padded (default ``4``).
        Use ``5`` for books whose pages are named like ``"00126r"``.
    folio_recto_suffix:
        When ``True``, the recto/verso suffix is included in the page name:
        folios with a trailing ``"v"``/``"V"`` get suffix ``"v"``; all
        others (plain numbers or explicit ``"r"``/``"R"``) get suffix ``"r"``.
        Example: folio ``"126"``  → ``"00126r"``,
                 folio ``"126v"`` → ``"00126v"``.
        When ``False`` (default), no suffix is appended — matching the
        original behaviour.

    Returns ``None`` if the folio cannot be resolved.
    """
    if folio_map is not None:
        return folio_map.get(str(folio), None)

    # Strip common manuscript folio prefixes: "f. ", "fol. ", "f.", "fol."
    # e.g. "f. 87v" → "87v",  "fol. 12" → "12"
    folio_clean = re.sub(r'^f(?:ol)?\.?\s*', '', folio.strip(), flags=re.IGNORECASE)

    # Determine verso/recto suffix from the trailing character
    suffix = ""
    if folio_recto_suffix:
        suffix = "v" if folio_clean.endswith(("v", "V")) else "r"

    # Strip trailing recto/verso marker before numeric conversion
    folio_num_str = folio_clean.rstrip("vrVR")
    try:
        page_idx = int(folio_num_str) + folio_offset
        return str(page_idx).zfill(folio_zfill) + suffix
    except ValueError:
        return None


def _load_pcgts(pcgts_dir: str, page_name: str) -> Optional[dict]:
    path = os.path.join(pcgts_dir, page_name, "pcgts.json")
    if not os.path.exists(path):
        # Also try without leading zeros
        path_plain = os.path.join(pcgts_dir, page_name.lstrip("0") or "0", "pcgts.json")
        if os.path.exists(path_plain):
            path = path_plain
        else:
            return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Mapping overview
# ---------------------------------------------------------------------------

def _print_mapping_overview(
    manuscript,
    folio_offset: int,
    folio_map: Optional[Dict[str, str]],
    page_index: Optional[Dict[str, Any]],   # DatabaseBook path
    pcgts_dir: Optional[str],               # raw dir path
    folio_zfill: int = 4,
    folio_recto_suffix: bool = False,
) -> None:
    """
    Print a human-readable table that shows, for every folio that appears in
    the monodi manuscript, what PCGTS page it resolves to and whether that
    page exists in the configured source.

    Also lists pages present in the source that are not referenced by any
    monodi folio.
    """
    # Collect all (folio, chant_id) pairs in document order, deduplicating
    # folio strings while preserving first occurrence for display.
    folio_chants: Dict[str, List[str]] = {}   # folio → list of chant short ids
    for chant in manuscript.chants:
        for pl in chant.page_lines:
            folio = str(pl.folio)
            folio_chants.setdefault(folio, [])
            chant_short = chant.meta.id[:8] + "…"
            if chant_short not in folio_chants[folio]:
                folio_chants[folio].append(chant_short)

    # All pages available in source
    all_source_pages: set = set()
    if page_index is not None:
        all_source_pages = set(page_index.keys())
    elif pcgts_dir and os.path.isdir(pcgts_dir):
        all_source_pages = {
            e.name for e in os.scandir(pcgts_dir)
            if e.is_dir() and os.path.exists(os.path.join(e.path, "pcgts.json"))
        }

    # Build rows
    rows: List[Dict] = []
    referenced_pages: set = set()
    for folio, chant_ids in folio_chants.items():
        page_name = _resolve_folio(folio, folio_offset, folio_map, folio_zfill, folio_recto_suffix)
        found: Optional[bool] = None
        if page_name is not None:
            if page_index is not None:
                found = page_name in page_index
            elif pcgts_dir:
                # check filesystem
                p = os.path.join(pcgts_dir, page_name, "pcgts.json")
                p2 = os.path.join(pcgts_dir, page_name.lstrip("0") or "0", "pcgts.json")
                found = os.path.exists(p) or os.path.exists(p2)
            if page_name:
                referenced_pages.add(page_name)
        rows.append({
            "folio":     folio,
            "page_name": page_name,
            "found":     found,
            "chants":    ", ".join(chant_ids[:3]) + ("…" if len(chant_ids) > 3 else ""),
        })

    unreferenced = sorted(all_source_pages - referenced_pages)

    # --- Print ---
    sep = "-" * 72
    print()
    print("Page mapping overview")
    print("=" * 72)
    src_label = (
        f"DatabaseBook '{manuscript.meta.id}'"
        if page_index is not None
        else f"pcgts-dir '{pcgts_dir}'"
    )
    print(f"  Monodi manuscript : {manuscript.meta.id}  ({len(manuscript.chants)} chants)")
    print(f"  PCGTS source      : {src_label}")
    if folio_map:
        print(f"  Folio map         : {len(folio_map)} explicit entries  (offset/zfill/suffix ignored)")
    else:
        suffix_label = "r/v appended" if folio_recto_suffix else "none"
        print(f"  Folio offset      : {folio_offset}   zfill: {folio_zfill}   suffix: {suffix_label}")
    print()
    print(f"  {'Folio':<12}  {'→  PCGTS page':<16}  {'Status':<18}  Chants (first 3)")
    print(f"  {sep}")
    found_count = missing_count = unresolvable_count = 0
    for row in rows:
        folio_s     = row["folio"]
        page_s      = row["page_name"] if row["page_name"] else "(unresolvable)"
        if row["found"] is True:
            status = "✓  found"
            found_count += 1
        elif row["found"] is False:
            status = "✗  not found"
            missing_count += 1
        else:
            status = "—  unresolvable"
            unresolvable_count += 1
        print(f"  {folio_s:<12}  {'→  ' + page_s:<16}  {status:<18}  {row['chants']}")

    print(f"  {sep}")
    print(f"  {found_count} found   {missing_count} missing   {unresolvable_count} unresolvable"
          f"   (out of {len(rows)} unique folios)")

    if unreferenced:
        print()
        print(f"  Pages in source not referenced by any monodi folio ({len(unreferenced)}):")
        # Print in rows of 10
        chunk = 10
        for i in range(0, len(unreferenced), chunk):
            print("    " + "  ".join(unreferenced[i:i + chunk]))

    print()


# ---------------------------------------------------------------------------
# Per-chant processing
# ---------------------------------------------------------------------------

def _process_chant(
    chant: MonodiChant,
    chant_folder: str,
    output_chant_folder: str,
    enricher: MonodiPositionEnricher,
    verbose: bool,
    # --- DatabaseBook path (preferred) ---
    page_index: Optional[Dict[str, Any]] = None,
    # --- raw PCGTS dir path (fallback) ---
    pcgts_dir: Optional[str] = None,
    folio_offset: int = 0,
    folio_map: Optional[Dict[str, str]] = None,
    folio_zfill: int = 4,
    folio_recto_suffix: bool = False,
) -> None:
    data_path = os.path.join(chant_folder, "data.json")
    with open(data_path, "r", encoding="utf-8") as fh:
        data_json = json.load(fh)

    folios_present = list(dict.fromkeys(pl.folio for pl in chant.page_lines))

    if page_index is not None:
        # ---- DatabaseBook path ----------------------------------------
        folio_to_db_page: Dict[str, Any] = {}
        for folio in folios_present:
            page_name = _resolve_folio(folio, folio_offset, folio_map, folio_zfill, folio_recto_suffix)
            if page_name is None:
                if verbose:
                    print(f"    ⚠  Folio '{folio}' — could not resolve to a page name; skipped")
                continue
            db_page = page_index.get(page_name)
            if db_page is None:
                if verbose:
                    print(f"    ⚠  Folio '{folio}' → page '{page_name}' — not found in book; skipped")
                continue
            folio_to_db_page[folio] = db_page
            if verbose:
                print(f"    Folio '{folio}' → page '{page_name}'  ✓")

        if not folio_to_db_page:
            print(f"    No matching pages found for any folio; skipping chant.")
            return

        enriched = enricher.enrich_full_chant(chant, data_json, folio_to_db_page=folio_to_db_page)

    else:
        # ---- Raw PCGTS directory path ----------------------------------
        folio_to_pcgts: Dict[str, dict] = {}
        for folio in folios_present:
            page_name = _resolve_folio(folio, folio_offset, folio_map, folio_zfill, folio_recto_suffix)
            if page_name is None:
                if verbose:
                    print(f"    ⚠  Folio '{folio}' — could not resolve to a page name; skipped")
                continue
            pcgts = _load_pcgts(pcgts_dir, page_name)
            if pcgts is None:
                if verbose:
                    print(f"    ⚠  Folio '{folio}' → page '{page_name}' — pcgts.json not found; skipped")
                continue
            folio_to_pcgts[folio] = pcgts
            if verbose:
                print(f"    Folio '{folio}' → page '{page_name}'  ✓")

        if not folio_to_pcgts:
            print(f"    No PCGTS pages found for any folio; skipping chant.")
            return

        enriched = enricher.enrich_full_chant(chant, data_json, folio_to_pcgts_json=folio_to_pcgts)

    stats = alignment_stats(enriched)
    pct = 100.0 * stats["matched"] / stats["total"] if stats["total"] else 0.0
    print(
        f"    Notes: {stats['total']} total | "
        f"{stats['matched']} matched ({pct:.0f}%) | "
        f"{stats['interpolated']} interpolated | "
        f"{stats['unpositioned']} unpositioned"
    )

    os.makedirs(output_chant_folder, exist_ok=True)
    out_data_path = os.path.join(output_chant_folder, "data.json")
    with open(out_data_path, "w", encoding="utf-8") as fh:
        json.dump(enriched, fh, indent=2, ensure_ascii=False)

    # Copy meta.json unchanged
    src_meta = os.path.join(chant_folder, "meta.json")
    dst_meta = os.path.join(output_chant_folder, "meta.json")
    shutil.copy2(src_meta, dst_meta)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich monodi+ export files with position data from PCGTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--monodi-path", required=True,
        help="Path to the manuscript export folder (contains meta.json + chant sub-folders).",
    )

    # PCGTS source — mutually exclusive
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--book",
        help="Name of the ommr4all DatabaseBook (preferred). "
             "Django is initialised automatically so pitches are resolved from clef context.",
    )
    src_group.add_argument(
        "--pcgts-dir",
        help="Path to the ommr4all book directory (fallback). "
             "PCGTS files are expected at <pcgts-dir>/<page>/pcgts.json.",
    )

    parser.add_argument(
        "--output-path", required=True,
        help="Output folder for enriched data.json files.",
    )
    parser.add_argument(
        "--folio-offset", type=int, default=0,
        help="Add this integer to the folio number to get the PCGTS page index (default: 0).",
    )
    parser.add_argument(
        "--folio-zfill", type=int, default=4, metavar="N",
        help="Zero-pad width for the numeric part of the resolved page name (default: 4). "
             "Use 5 for books with names like '00126r'.",
    )
    parser.add_argument(
        "--folio-recto-suffix", action="store_true",
        help="Append 'r' or 'v' to the resolved page name. "
             "Folio '126' → '00126r', folio '126v' → '00126v'. "
             "Required for books whose page names include a recto/verso suffix.",
    )
    parser.add_argument(
        "--folio-map", default=None,
        help='JSON object mapping folio IDs to page names, e.g. \'{"86": "0085", "86v": "0086"}\'.',
    )
    parser.add_argument(
        "--include-note-types", nargs="*", default=None,
        metavar="TYPE",
        help="Only align notes with these noteType values (default: all). "
             "E.g.: Normal Liquescent Oriscus Strophicus",
    )
    parser.add_argument(
        "--show-mapping", action="store_true",
        help="Print a table of folio → PCGTS-page mappings and exit without enriching.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print folio resolution details for each chant.",
    )
    args = parser.parse_args()

    # Parse folio map
    folio_map: Optional[Dict[str, str]] = None
    if args.folio_map:
        try:
            folio_map = json.loads(args.folio_map)
        except json.JSONDecodeError as exc:
            parser.error(f"--folio-map is not valid JSON: {exc}")

    # Set up DatabaseBook page index when --book is used
    page_index: Optional[Dict[str, Any]] = None
    if args.book:
        print(f"Initialising Django and loading book '{args.book}' …")
        _setup_django()
        page_index = _build_page_index(args.book)
        print(f"  {len(page_index)} pages found in book '{args.book}'")

    # Load manuscript
    print(f"Loading manuscript from: {args.monodi_path}")
    manuscript = load_manuscript(args.monodi_path)
    print(f"  '{manuscript.meta.id}' — {len(manuscript.chants)} chants")

    # Mapping overview (always shown; --show-mapping exits afterwards)
    _print_mapping_overview(
        manuscript=manuscript,
        folio_offset=args.folio_offset,
        folio_map=folio_map,
        page_index=page_index,
        pcgts_dir=args.pcgts_dir,
        folio_zfill=args.folio_zfill,
        folio_recto_suffix=args.folio_recto_suffix,
    )
    if args.show_mapping:
        return

    # Create output directory and copy manuscript meta
    os.makedirs(args.output_path, exist_ok=True)
    shutil.copy2(
        os.path.join(args.monodi_path, "meta.json"),
        os.path.join(args.output_path, "meta.json"),
    )

    enricher = MonodiPositionEnricher(include_note_types=args.include_note_types)

    for chant in manuscript.chants:
        chant_id     = chant.meta.id
        chant_folder = os.path.join(args.monodi_path, chant_id)
        out_folder   = os.path.join(args.output_path, chant_id)

        n_lines = len(chant.page_lines)
        print(f"\n  [{chant.meta.textinitium}]  {chant_id[:8]}…  "
              f"folio {chant.meta.foliostart}, line {chant.meta.zeilenstart}  "
              f"({n_lines} page-lines)")

        if not os.path.isdir(chant_folder):
            print(f"    Chant folder not found: {chant_folder}; skipped.")
            continue

        _process_chant(
            chant=chant,
            chant_folder=chant_folder,
            output_chant_folder=out_folder,
            enricher=enricher,
            verbose=args.verbose,
            page_index=page_index,
            pcgts_dir=args.pcgts_dir,
            folio_offset=args.folio_offset,
            folio_map=folio_map,
            folio_zfill=args.folio_zfill,
            folio_recto_suffix=args.folio_recto_suffix,
        )

    print(f"\nDone. Enriched files written to: {args.output_path}")


if __name__ == "__main__":
    main()
