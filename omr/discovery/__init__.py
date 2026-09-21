"""Experimental embedding workflows for symbol annotation.

The page-suggestion path describes symbol-bearing foreground with self-supervised patch
embeddings and recommends a diverse batch of pages to correct before supervised fine-tuning. It
does not localise or classify symbols.

The separate discovery prototype proposes symbol candidates, clusters them for human review,
groups them into neumes and measures graphical localisation, semantic interpretation, grouping
and neume interpretation independently.

Nothing here writes into a book's `pcgts.json`. Experimental output lives in a run directory;
the existing symbol training operation consumes pages only after users correct them and set their
`Symbols` progress lock.

Entry point: `python -m omr.discovery.cli --help`.
"""
