"""Interchangeable clustering of symbol and neume embeddings.

Cluster ids are similarity labels only. No clusterer reads or writes `SymbolLabel` or
`NeumeCandidate.neume_type`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np

from omr.discovery.config import ClusterConfig


@dataclass
class ClusterResult:
    labels: List[str]
    method: str
    params: Dict[str, Any]
    seed: int
    n_clusters: int
    n_outliers: int


class Clusterer(ABC):
    name = 'base'

    @abstractmethod
    def fit_predict(self, x: np.ndarray, cfg: ClusterConfig) -> List[int]: ...


class HdbscanClusterer(Clusterer):
    name = 'hdbscan'

    def fit_predict(self, x, cfg):
        from sklearn.cluster import HDBSCAN
        if len(x) < 2:
            return [-1] * len(x)
        size = min(max(2, cfg.min_cluster_size), len(x))
        return HDBSCAN(min_cluster_size=size).fit_predict(x).astype(int).tolist()


class AgglomerativeClusterer(Clusterer):
    name = 'agglomerative'

    def fit_predict(self, x, cfg):
        from sklearn.cluster import AgglomerativeClustering
        if len(x) < 2:
            return [0] * len(x)
        if cfg.distance_threshold is not None:
            model = AgglomerativeClustering(n_clusters=None,
                                            distance_threshold=cfg.distance_threshold)
        else:
            model = AgglomerativeClustering(n_clusters=min(cfg.n_clusters, len(x)))
        return model.fit_predict(x).astype(int).tolist()


class KMeansClusterer(Clusterer):
    name = 'kmeans'

    def fit_predict(self, x, cfg):
        from sklearn.cluster import KMeans
        if len(x) < 2:
            return [0] * len(x)
        return KMeans(n_clusters=min(cfg.n_clusters, len(x)), random_state=cfg.seed,
                      n_init=10).fit_predict(x).astype(int).tolist()


CLUSTERERS: Dict[str, Type[Clusterer]] = {
    c.name: c for c in (HdbscanClusterer, AgglomerativeClusterer, KMeansClusterer)
}


def _geometry(candidate) -> np.ndarray:
    # Geometry is scale-free even without reloading PCGTS: aspect ratio and log area describe
    # shape while appearance remains the default (`geometry_weight == 0`).
    eps = 1e-12
    return np.array([np.log(max(candidate.box.w, eps)), np.log(max(candidate.box.h, eps)),
                     np.log(max(candidate.box.w, eps) / max(candidate.box.h, eps))], dtype=np.float32)


def build_features(store, candidates, cfg: ClusterConfig, *, neumes: bool = False) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 0), dtype=np.float32)
    source = store.neume_embeddings if neumes else store.embeddings
    if source is None:
        raise ValueError('the store has no {} embeddings'.format('neume' if neumes else 'symbol'))
    rows = []
    for candidate in candidates:
        if candidate.embedding_index is None:
            raise ValueError('{} has no embedding index'.format(candidate.id))
        rows.append(source[candidate.embedding_index])
    x = np.asarray(rows, dtype=np.float32)

    if cfg.pca_dim is not None and len(x) > 1 and x.shape[1] > 1:
        from sklearn.decomposition import PCA
        n_components = min(cfg.pca_dim, len(x) - 1, x.shape[1])
        if n_components > 0 and n_components < x.shape[1]:
            x = PCA(n_components=n_components, random_state=cfg.seed).fit_transform(x)
    if cfg.geometry_weight:
        geometry = np.stack([_geometry(c) for c in candidates]) * cfg.geometry_weight
        x = np.concatenate([x, geometry], axis=1)
    return x.astype(np.float32)


def _cluster(store, candidates, cfg: ClusterConfig, prefix: str, *, neumes: bool) -> ClusterResult:
    if cfg.method not in CLUSTERERS:
        raise ValueError('unknown clusterer {!r} (known: {})'.format(
            cfg.method, ', '.join(sorted(CLUSTERERS))))
    if not candidates:
        return ClusterResult([], cfg.method, cfg.to_dict(), cfg.seed, 0, 0)
    x = build_features(store, candidates, cfg, neumes=neumes)
    labels_int = CLUSTERERS[cfg.method]().fit_predict(x, cfg)
    labels = [prefix + str(label) for label in labels_int]
    for candidate, label in zip(candidates, labels):
        candidate.cluster_id = label
    clusters = {label for label in labels_int if label != -1}
    return ClusterResult(labels, cfg.method, cfg.to_dict(), cfg.seed, len(clusters),
                         sum(label == -1 for label in labels_int))


def cluster_candidates(store, candidates, cfg: ClusterConfig, prefix: str = '') -> ClusterResult:
    return _cluster(store, candidates, cfg, prefix, neumes=False)


def cluster_neumes(store, cfg: ClusterConfig, prefix: str = '') -> ClusterResult:
    return _cluster(store, store.ordered_neumes(), cfg, prefix, neumes=True)


def recluster(store, candidate_ids: List[str], cfg: ClusterConfig) -> ClusterResult:
    candidates = [store.candidates[cid] for cid in candidate_ids]
    parents = {c.cluster_id for c in candidates}
    if len(parents) != 1:
        raise ValueError('recursive refinement requires candidates from one parent cluster')
    return cluster_candidates(store, candidates, cfg, prefix=next(iter(parents)) + '.')
