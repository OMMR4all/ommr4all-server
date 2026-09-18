"""Experimental self-supervised symbol and neume discovery.

This package is an *offline* prototype: it reads a book from the storage, proposes symbol
candidates on every music line without any supervision, clusters them for human review,
groups them into neumes and measures all four concerns (graphical symbol localisation,
semantic symbol interpretation, grouping into neumes, semantic neume interpretation)
separately.

Nothing here touches the supervised pipeline in `omr/steps/symboldetection` and nothing
here writes into a book's `pcgts.json`. Candidates live in their own run directory
(see `omr.discovery.store`); converting an accepted candidate set into ground truth is a
later, deliberately separate step.

Entry point: `python -m omr.discovery.cli --help`.
"""
