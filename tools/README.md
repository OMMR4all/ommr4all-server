# `tools/`

Standalone helper scripts around the OMMR4all server: dataset exports, dictionary
generation, evaluation drivers and the data crawlers. They are **not** part of the
web application and are never imported by it, with the two exceptions marked
below.

Run them from this directory's parent (the server module) as a module, so the
`tools` package resolves:

```bash
cd modules/ommr4all-server
python -m tools.export.export_dataset_no_images --help
```

Most scripts set `DJANGO_SETTINGS_MODULE` and call `django.setup()` themselves.
Scripts still carrying hard coded book names or paths have them as `argparse`
defaults or as a constant at the top of the file.

> This directory was cleaned up on 2026-10-20: about 60 one-off scripts from
> finished campaigns (Rom_1, Mulhouse 2, Assisi, Graduale Synopticum, Geesebook
> fixes, the guppyocr cross validation) were removed. They are still in the git
> history, so any of them can be brought back with
> `git log --diff-filter=D --name-only -- tools/` followed by
> `git show <commit>:<path>`.

## Do not delete: imported by the server

| File | Imported by |
|---|---|
| `simple_gregorianik_text_export.py` | defines `Lyrics` / `Lyric_info` / `Metainfo` / `Variant`; used by `database/start_up/load_text_variants_in_memory.py`, `database/file_formats/book/documents.py`, `omr/steps/text/text_localisation/predictor.py`, `omr/steps/text/correction_tools/document_matching_corrector/predictor.py`, `omr/experimenter/documents/doc_instace_evaluator.py`. Running it directly (`crawl()`) re-crawls gregorianik.uni-regensburg.de. |
| `dataset_statistics.py` | `drop_captial_of_text_line_center`, `para_text_of_text_line_center` and the `Counts` statistics used by `omr/steps/symboldetection/evaluator.py`. |

Both belong in `database/` rather than in `tools/`; moving them means touching the
five importing modules and has not been done yet.

## Dataset export

| Script | What it does |
|---|---|
| `export/export_dataset_no_images.py` | **Ground-truth dataset without the page images.** Writes `pcgts.json` + `page_progress.json` + `meta.json` per verified page into `<out>/<book>/<page>/`. This is how the published GT datasets are built. |
| `export/export_data.py` | LLM fine-tuning dataset of whole pages. Without `--with-images` only the text transcription (`<line>…</line>` with the `(Type\|Pos\|GC)` symbol encoding) is written; with it, an unsloth-style `train_data.jsonl` plus a copy of every page image. |
| `export/export_data_line.py` | Same encoding, but one cropped image per music block together with its connected lyric regions. Has no CLI yet; book list and output path are in its `__main__` block. |
| `export/export_pages_for_ocr.py` | Image + `.txt` pairs of randomly picked pages, named by UUID, as OCR training data. Book list in the `__main__` block. |

```bash
# annotations only, no images
python -m tools.export.export_dataset_no_images --books mul_2_rsync_gt --out /tmp/gt

# text-only LLM dataset
python -m tools.export.export_data --books Geesebook1_complete_fixed_ro --out /tmp/llm

# the same including the page images
python -m tools.export.export_data --books Geesebook1_complete_fixed_ro --out /tmp/llm --with-images
```

## Corpus Monodicum export

| Script | What it does |
|---|---|
| `export/export_docs.py` | Documents of a book as the CM metadata xlsx. `--skip-flagged` leaves out documents whose text still contains the flag character, `--write-skipped` puts those into a second file. |
| `export/export_mdocs_specified_by_xlms.py` | Monodi JSON (`data.json` + `meta.json` per document) for the documents listed in such an xlsx. |

```bash
python -m tools.export.export_docs --book Rom_1_export --skip-flagged
python -m tools.export.export_mdocs_specified_by_xlms --book Geesebook2_andreas \
    --source Geesebook2 --xlsx "/tmp/CM Default Metadatendatei.xlsx" --out /tmp/export
```

## Evaluation

| Script | What it does |
|---|---|
| `export/evaluate_documents.py` | Document- and line-level evaluation of a prediction book against a GT book (layout, symbols, syllables, stafflines) into one xlsx. The only entry point into the `eval_*` helpers of `omr/experimenter/documents/`. Books and the document-filter xlsx files are constants in its `__main__` block. |
| `export/evaluate_llm_output.py` | Parses the `<line>…</line>` format produced by `export/export_data.py` back into symbols and text and scores a model's predictions against the GT jsonl. Paths are constants in its `__main__` block. |
| `evaluate_timing.py` | Aggregates the per-page tool timings (`page_statistics`) of a book, also from a zipped book export. |
| `book_statistics_cli.py` | `prettytable` overview of a book: pages, lines, symbols, syllables. Supports `--ignore-page`, `--specific-pages-only`, `--below-specific-page`. |

```bash
python -m tools.book_statistics_cli --books mul_2_rsync_gt
```

Both evaluation scripts write legacy `.xls` through **`xlwt`, which is not a
declared dependency of the server** (and is abandoned upstream — despite the
`.xlsx` file names it produces the old `.xls` format). Install it into the venv
before running them:

```bash
uv pip install xlwt
```

Porting them to `openpyxl` (which *is* a dependency) also requires changing the
`sheet.write(row, col, value)` calls in `omr/experimenter/documents/doc_instace_evaluator.py`.

## Dictionaries and lyric sources

| Script | What it does |
|---|---|
| `create_hyphenation_dictionary.py` | Hyphenation dictionary from the GT lyrics of the given books. `--books … --output …` |
| `word_frequency_list.py` | Word frequency list, bigrams and a sentence file from a Monodi export directory. Path in the `__main__` block. |
| `generate_abbreviations.py` | Abbreviation/acronym list from a CantusCorpus `chant.csv`. Path in the `__main__` block. |
| `gregorien_info_extracter.py` | Crawls gregorien.info and appends the result to `internal_storage/resources/lyrics_collection/lyrics_by_sources.json`, the lyrics database the server loads at start-up. Takes hours and hits ~8800 URLs. |

## Lyrics by Gemini

| Script | What it does |
|---|---|
| `gemini_batch_lyrics.py` | Transcribes the lyrics of a page range with the Gemini Batch API, resumable across crashes via a state file, and can write the result back into the PcGts files. See the module docstring for the full `submit` / `status` / `fetch` / `resubmit` / `apply` workflow. Needs the optional `google-genai` package. |

## PcGts repair

| Script | What it does |
|---|---|
| `staff_lines/staffline_shifter.py` | `shift_staff_lines_by_pixels(pages, delta_y)` moves every staff line of the given pages up or down by a number of pixels and **overwrites the PcGts files**. Import it and pass the pages; it has no CLI. |
