"""The run directory: candidates, neumes, embeddings, interactions and the run record.

Layout of `<out_root>/<run_id>/`::

    run.json               RunRecord (config, environment, timings, counts)
    candidates.json        {"schema_version", "run_id", "candidates": [...]}
    embeddings.npy         float32 (N, D), row `SymbolCandidate.embedding_index`
    neumes.json            {"schema_version", "grouping_run_id", "neumes": [...]}
    neume_embeddings.npy   float32 (M, D)
    interactions.jsonl     one `Interaction` per line
    metrics.json           evaluation output
    report.md              human readable report
    overlays/<page>.jpg    candidate overlay on color_highres_preproc
    clusters/cluster_<id>.jpg
    neume_overlays/<page>.jpg
    neume_clusters/cluster_<id>.jpg

Candidates deliberately live here and not in the book's `pcgts.json`: they are proposals of
an experimental pipeline, they carry review state that PCGTS cannot express, and writing them
into the page would mix them with ground truth. Converting an accepted set into ground truth
is a separate, explicit step.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from mashumaro.mixins.json import DataClassJSONMixin

from database.file_write import write_text_atomic
from omr.discovery.provenance import RunRecord
from omr.discovery.schema import (GroupingState, NeumeCandidate, SymbolCandidate, validate_label,
                                  validate_neume_type)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

RUN_FILE = 'run.json'
CANDIDATES_FILE = 'candidates.json'
EMBEDDINGS_FILE = 'embeddings.npy'
NEUMES_FILE = 'neumes.json'
NEUME_EMBEDDINGS_FILE = 'neume_embeddings.npy'
INTERACTIONS_FILE = 'interactions.jsonl'
METRICS_FILE = 'metrics.json'
REPORT_FILE = 'report.md'
OVERLAY_DIR = 'overlays'
CLUSTER_DIR = 'clusters'
NEUME_OVERLAY_DIR = 'neume_overlays'
NEUME_CLUSTER_DIR = 'neume_clusters'


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Interaction(DataClassJSONMixin):
    """One human (or, offline, one scripted) review action.

    The raw material of the annotation efficiency metric: how many labels does one action set?
    """
    ts: str
    op: str
    #: 'cluster:<id>' | 'selection' | 'instance:<id>' | 'group:<id>' | 'auto:<name>'
    scope: str
    n_affected: int
    detail: Dict[str, str] = field(default_factory=dict)


class SchemaVersionMismatch(Exception):
    def __init__(self, path: str, found: Any):
        super().__init__('{} has schema version {!r}, expected {}'.format(path, found, SCHEMA_VERSION))
        self.path = path
        self.found = found


class CandidateStore:
    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, run_dir: str, run: RunRecord):
        self.run_dir = run_dir
        self.run = run
        self.candidates: Dict[str, SymbolCandidate] = {}
        self.neumes: Dict[str, NeumeCandidate] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.neume_embeddings: Optional[np.ndarray] = None
        self.interactions: List[Interaction] = []

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def create(cls, run_dir: str, run: RunRecord) -> 'CandidateStore':
        os.makedirs(run_dir, exist_ok=True)
        for sub in (OVERLAY_DIR, CLUSTER_DIR):
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        return cls(run_dir, run)

    @classmethod
    def load(cls, run_dir: str) -> 'CandidateStore':
        with open(os.path.join(run_dir, RUN_FILE)) as f:
            run_dict = json.load(f)
        if run_dict.get('schema_version') != SCHEMA_VERSION:
            raise SchemaVersionMismatch(os.path.join(run_dir, RUN_FILE), run_dict.get('schema_version'))
        store = cls(run_dir, RunRecord.from_dict(run_dict))

        candidates_path = os.path.join(run_dir, CANDIDATES_FILE)
        if os.path.exists(candidates_path):
            with open(candidates_path) as f:
                payload = json.load(f)
            if payload.get('schema_version') != SCHEMA_VERSION:
                raise SchemaVersionMismatch(candidates_path, payload.get('schema_version'))
            for d in payload.get('candidates', []):
                c = SymbolCandidate.from_dict(d)
                store.candidates[c.id] = c

        neumes_path = os.path.join(run_dir, NEUMES_FILE)
        if os.path.exists(neumes_path):
            with open(neumes_path) as f:
                payload = json.load(f)
            if payload.get('schema_version') != SCHEMA_VERSION:
                raise SchemaVersionMismatch(neumes_path, payload.get('schema_version'))
            for d in payload.get('neumes', []):
                n = NeumeCandidate.from_dict(d)
                store.neumes[n.id] = n

        embeddings_path = os.path.join(run_dir, EMBEDDINGS_FILE)
        if os.path.exists(embeddings_path):
            store.embeddings = np.load(embeddings_path)
        neume_embeddings_path = os.path.join(run_dir, NEUME_EMBEDDINGS_FILE)
        if os.path.exists(neume_embeddings_path):
            store.neume_embeddings = np.load(neume_embeddings_path)

        interactions_path = os.path.join(run_dir, INTERACTIONS_FILE)
        if os.path.exists(interactions_path):
            with open(interactions_path) as f:
                store.interactions = [Interaction.from_dict(json.loads(line))
                                      for line in f if line.strip()]
        return store

    # -- persistence ----------------------------------------------------------------------
    def save(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        run_dict = self.run.to_dict()
        run_dict['schema_version'] = SCHEMA_VERSION
        write_text_atomic(os.path.join(self.run_dir, RUN_FILE), json.dumps(run_dict, indent=2))

        write_text_atomic(os.path.join(self.run_dir, CANDIDATES_FILE), json.dumps({
            'schema_version': SCHEMA_VERSION,
            'run_id': self.run.run_id,
            'candidates': [c.to_dict() for c in self.ordered_candidates()],
        }, indent=1))

        if self.neumes:
            write_text_atomic(os.path.join(self.run_dir, NEUMES_FILE), json.dumps({
                'schema_version': SCHEMA_VERSION,
                'grouping_run_id': self.run.run_id,
                'symbols_run_id': self.run.parent_run_id,
                'neumes': [n.to_dict() for n in self.ordered_neumes()],
            }, indent=1))

        if self.embeddings is not None:
            np.save(os.path.join(self.run_dir, EMBEDDINGS_FILE), self.embeddings.astype(np.float32))
        if self.neume_embeddings is not None:
            np.save(os.path.join(self.run_dir, NEUME_EMBEDDINGS_FILE), self.neume_embeddings.astype(np.float32))

        if self.interactions:
            write_text_atomic(os.path.join(self.run_dir, INTERACTIONS_FILE),
                              ''.join(json.dumps(i.to_dict()) + '\n' for i in self.interactions))

    def write_json(self, name: str, payload: Any) -> str:
        path = os.path.join(self.run_dir, name)
        write_text_atomic(path, json.dumps(payload, indent=2))
        return path

    def write_text(self, name: str, content: str) -> str:
        path = os.path.join(self.run_dir, name)
        write_text_atomic(path, content)
        return path

    def sub_dir(self, name: str) -> str:
        path = os.path.join(self.run_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    # -- candidates -----------------------------------------------------------------------
    def add_candidates(self, candidates: List[SymbolCandidate], embeddings: np.ndarray) -> None:
        """Append candidates and their embeddings, keeping `embedding_index` consistent."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(candidates):
            raise ValueError('expected one embedding row per candidate, got {} rows for {} candidates'
                             .format(embeddings.shape, len(candidates)))
        offset = 0 if self.embeddings is None else self.embeddings.shape[0]
        for i, c in enumerate(candidates):
            c.embedding_index = offset + i
            self.candidates[c.id] = c
        self.embeddings = embeddings if self.embeddings is None \
            else np.concatenate([self.embeddings, embeddings], axis=0)

    def set_candidates(self, candidates: List[SymbolCandidate], embeddings: np.ndarray) -> None:
        self.candidates = {}
        self.embeddings = None
        self.add_candidates(candidates, embeddings)

    def embedding_of(self, candidate: SymbolCandidate) -> np.ndarray:
        if self.embeddings is None or candidate.embedding_index is None:
            raise ValueError('candidate {} has no embedding'.format(candidate.id))
        return self.embeddings[candidate.embedding_index]

    def neume_embedding_of(self, neume: NeumeCandidate) -> np.ndarray:
        if self.neume_embeddings is None or neume.embedding_index is None:
            raise ValueError('neume {} has no embedding'.format(neume.id))
        return self.neume_embeddings[neume.embedding_index]

    def ordered_candidates(self) -> List[SymbolCandidate]:
        # UUID is only the final tie breaker: candidate ordering, PCA and KMeans must not depend
        # on freshly generated ids when two components share the same x centre.
        return sorted(self.candidates.values(), key=lambda c: (
            c.page, c.line_id, c.center_x, c.center_y, c.box.w, c.box.h,
            c.discovery_method, c.id))

    def ordered_neumes(self) -> List[NeumeCandidate]:
        return sorted(self.neumes.values(), key=lambda n: (
            n.page, n.line_id, n.box.x, n.box.y, n.box.w, n.box.h,
            tuple(n.component_ids), n.id))

    def candidates_of_page(self, page: str) -> List[SymbolCandidate]:
        return [c for c in self.ordered_candidates() if c.page == page]

    def candidates_of_line(self, page: str, line_id: str) -> List[SymbolCandidate]:
        return [c for c in self.ordered_candidates() if c.page == page and c.line_id == line_id]

    def neumes_of_page(self, page: str) -> List[NeumeCandidate]:
        return [n for n in self.ordered_neumes() if n.page == page]

    def pages(self) -> List[str]:
        return sorted({c.page for c in self.candidates.values()} | {n.page for n in self.neumes.values()})

    def log_interaction(self, interaction: Interaction) -> None:
        self.interactions.append(interaction)

    # -- invariants -----------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Every violated invariant as a human readable line; empty means consistent."""
        errors: List[str] = []
        claimed_by: Dict[str, str] = {}

        for c in self.candidates.values():
            errors.extend('candidate {}: {}'.format(c.id, e) for e in validate_label(c.label))

        for n in self.ordered_neumes():
            errors.extend('neume {}: {}'.format(n.id, e) for e in validate_neume_type(n.neume_type))
            if not n.component_ids:
                errors.append('neume {}: has no components'.format(n.id))
            if len(set(n.component_ids)) != len(n.component_ids):
                errors.append('neume {}: duplicate component ids'.format(n.id))

            components = []
            for cid in n.component_ids:
                c = self.candidates.get(cid)
                if c is None:
                    errors.append('neume {}: component {} does not exist'.format(n.id, cid))
                    continue
                components.append(c)
                if cid in claimed_by and claimed_by[cid] != n.id:
                    errors.append('candidate {} is claimed by neume {} and neume {}'.format(
                        cid, claimed_by[cid], n.id))
                claimed_by[cid] = n.id
                if c.neume_id != n.id:
                    errors.append('candidate {} does not back-reference neume {} (neume_id={!r})'.format(
                        cid, n.id, c.neume_id))
                if (c.book,c.page,c.line_id) != (n.book,n.page,n.line_id):
                    errors.append('candidate {} is on {}/{}/{} but neume {} is on {}/{}/{}'.format(
                        cid,c.book,c.page,c.line_id,n.id,n.book,n.page,n.line_id))

            expected_relations = list(zip(n.component_ids, n.component_ids[1:]))
            actual_relations = [(r.from_id, r.to_id) for r in n.relations]
            if actual_relations != expected_relations:
                errors.append('neume {}: relations {} are not the consecutive chain {}'.format(
                    n.id, actual_relations, expected_relations))

            if components and n.grouping_state != GroupingState.MODIFIED:
                union = components[0].box
                for c in components[1:]:
                    union = union.union(c.box)
                if max(abs(union.x - n.box.x), abs(union.y - n.box.y),
                       abs(union.w - n.box.w), abs(union.h - n.box.h)) > 1e-9:
                    errors.append('neume {}: box is not the union of its components'.format(n.id))

        for c in self.candidates.values():
            if c.neume_id is not None and c.neume_id not in self.neumes:
                errors.append('candidate {}: neume_id {} does not exist'.format(c.id, c.neume_id))
            elif c.neume_id is not None and c.id not in self.neumes[c.neume_id].component_ids:
                errors.append('candidate {}: neume {} does not list it as a component'.format(c.id, c.neume_id))
        return errors
