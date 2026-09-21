"""Offline orchestration of page suggestions and symbol/neume discovery."""
import copy
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from database import DatabaseBook
from omr.discovery.clustering import cluster_candidates, cluster_neumes
from database.file_formats.performance.pageprogress import Locks
from database.file_write import write_text_atomic
from omr.discovery.config import RunConfig
from omr.discovery.discovery import DiscoveryContext, build_methods, nms, remove_staff_lines
from omr.discovery.evaluation import (annotation_efficiency, cluster_metrics, groundtruth_candidates,
                                      groundtruth_neumes, groundtruth_symbols, grouping_metrics,
                                      localization_metrics, write_report)
from omr.discovery.features import build_feature_extractor
from omr.discovery.grouping import build_grouping_method
from omr.discovery.page_selection import (
    PAGE_EMBEDDING_INDEX_FILE,
    PAGE_EMBEDDINGS_FILE,
    SUGGESTIONS_FILE,
    PageEmbeddingStats,
    foreground_page_embedding,
    select_representative_pages,
)
from omr.discovery.neume_embedding import embed_neumes
from omr.discovery.provenance import (RunRecord, StageTimer, collect_environment, peak_cuda_bytes,
                                      peak_rss_bytes, seed_everything)
from omr.discovery.regions import (count_music_lines, page_box_to_crop, staff_crops_of_page)
from omr.discovery.review import bulk_apply, reject
from omr.discovery.schema import (Box, MaskRle, NeumeCandidate, RejectionReason, ReviewState,
                                  SymbolCandidate, SymbolFamily, SymbolLabel)
from omr.discovery.store import (CLUSTER_DIR, METRICS_FILE, NEUME_CLUSTER_DIR, NEUME_OVERLAY_DIR,
                                 OVERLAY_DIR, RUN_FILE, SCHEMA_VERSION, CandidateStore, utc_now)
from omr.discovery.visualize import (cluster_contact_sheet, neume_contact_sheet,
                                     neume_page_overlay, page_overlay, safe_cluster_id)

logger = logging.getLogger(__name__)


def _run_id(cfg):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return '{}_{}_{}'.format(stamp,cfg.config_hash(),uuid4().hex[:8])


def _pages(cfg):
    book=DatabaseBook(cfg.book); available={p.page:p for p in book.pages()}
    names=cfg.pages or sorted(available)
    missing=[name for name in names if name not in available]
    if missing: raise ValueError('pages do not exist in book {}: {}'.format(cfg.book, ', '.join(missing)))
    return book,[available[name] for name in names]


def _record(cfg, run_id, kind, pages, parent=None, source=None):
    return RunRecord(schema_version=CandidateStore.SCHEMA_VERSION,run_id=run_id,kind=kind,
                     config=copy.deepcopy(cfg),created_at=utc_now(),parent_run_id=parent,
                     symbol_source=source,environment=collect_environment(),seed=cfg.seed,
                     pages=[p.page for p in pages])


def _finish(store):
    store.run.finished_at=utc_now(); store.run.peak_rss_bytes=peak_rss_bytes()
    store.run.peak_cuda_bytes=peak_cuda_bytes(); store.save()


def _feature_embedding(feature_map, proposal):
    return feature_map.pool_box(proposal.x,proposal.y,proposal.w,proposal.h).astype(np.float32)


def run_symbol_discovery(cfg: RunConfig, out_root: str,
                         extractor_override: Optional[str]=None) -> str:
    cfg=copy.deepcopy(cfg)
    if extractor_override: cfg.features.backend=extractor_override
    seed_everything(cfg.seed); book,pages=_pages(cfg)
    run_id=_run_id(cfg); run_dir=os.path.join(out_root,run_id)
    record=_record(cfg,run_id,'symbols',pages); timer=StageTimer(record)
    store=CandidateStore.create(run_dir,record)
    extractor=build_feature_extractor(cfg.features); methods=build_methods(cfg.discovery)
    record.n_lines_total=count_music_lines(pages)
    crops={}; candidates=[]; embeddings=[]
    with timer('load_features_and_discover'):
        extractor.load(); record.feature_extractor=extractor.describe()
        record.device=record.feature_extractor.get('device','')
        for db_page in pages:
            page_crops=staff_crops_of_page(db_page,cfg.crop)
            record.n_lines_dropped += len(db_page.pcgts().page.all_music_lines())-len(page_crops)
            mtime=os.stat(db_page.local_file_path('pcgts.json')).st_mtime_ns
            for crop in page_crops:
                crops[(crop.page,crop.line_id)]=crop
                feature_map=extractor.extract_feature_map(crop.image,crop.staff_space_px)
                ink=remove_staff_lines(crop.binary,crop.staff_lines_px,crop.staff_space_px,cfg.discovery)
                ctx=DiscoveryContext(crop=crop,cfg=cfg.discovery,ink=ink,feature_map=feature_map)
                proposals=nms(sum((method.discover(ctx) for method in methods),[]),cfg.discovery.nms_iou)
                for proposal in proposals:
                    box=__import__('omr.discovery.regions',fromlist=['crop_box_to_page']).crop_box_to_page(
                        crop,proposal.x,proposal.y,proposal.w,proposal.h)
                    cx,cy=box.center(); now=utc_now()
                    mask=MaskRle.from_array(proposal.mask) if proposal.mask is not None else None
                    candidates.append(SymbolCandidate(
                        id=uuid4().hex,run_id=run_id,book=cfg.book,page=crop.page,
                        block_id=crop.block_id,line_id=crop.line_id,box=box,center_x=cx,center_y=cy,
                        mask_rle=mask,discovery_score=proposal.score,
                        discovery_method=proposal.method,source_geometry_version='{}:{}'.format(mtime,crop.line_id),
                        created_at=now,updated_at=now))
                    embeddings.append(_feature_embedding(feature_map,proposal))
    if candidates: store.add_candidates(candidates,np.stack(embeddings))
    else: store.embeddings=np.zeros((0,record.feature_extractor.get('embedding_dim',0)),np.float32)
    with timer('cluster'):
        result=cluster_candidates(store,store.ordered_candidates(),cfg.clustering)
        record.counts.update(candidates=len(candidates),clusters=result.n_clusters,outliers=result.n_outliers)
    with timer('visualize'):
        for page in pages:
            page_overlay(page,store.candidates_of_page(page.page),
                         os.path.join(run_dir,OVERLAY_DIR,page.page+'.jpg'))
        for cluster_id in sorted({c.cluster_id for c in candidates}):
            cluster_contact_sheet(store,cluster_id,crops,os.path.join(
                run_dir,CLUSTER_DIR,'cluster_'+safe_cluster_id(cluster_id)+'.jpg'))
    _finish(store); run_evaluation(run_dir)
    return run_dir


def _pool_candidate_embeddings(candidates,crops,extractor):
    maps={}; rows=[]
    for index,c in enumerate(candidates):
        crop=crops[(c.page,c.line_id)]
        if (c.page,c.line_id) not in maps:
            maps[(c.page,c.line_id)]=extractor.extract_feature_map(crop.image,crop.staff_space_px)
        x,y,w,h=page_box_to_crop(crop,c.box)
        rows.append(maps[(c.page,c.line_id)].pool_box(x,y,w,h)); c.embedding_index=index
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0,0),np.float32)


def run_neume_grouping(cfg: RunConfig, symbols_run_dir: str, out_root: str,
                       symbol_source: str='discovered') -> str:
    if symbol_source not in ('groundtruth','discovered'): raise ValueError('invalid symbol source')
    cfg=copy.deepcopy(cfg); seed_everything(cfg.seed)
    source=CandidateStore.load(symbols_run_dir)
    cfg.book=source.run.config.book; cfg.pages=list(source.run.pages)
    book,pages=_pages(cfg); run_id=_run_id(cfg); run_dir=os.path.join(out_root,run_id)
    record=_record(cfg,run_id,'neumes',pages,source.run.run_id,symbol_source)
    store=CandidateStore.create(run_dir,record); timer=StageTimer(record)
    extractor=build_feature_extractor(cfg.features); extractor.load()
    record.feature_extractor=extractor.describe(); record.device=record.feature_extractor.get('device','')
    crops={}
    with timer('load_crops'):
        for page in pages:
            for crop in staff_crops_of_page(page,cfg.crop): crops[(crop.page,crop.line_id)]=crop
        record.n_lines_total=count_music_lines(pages); record.n_lines_dropped=record.n_lines_total-len(crops)
    with timer('prepare_symbols'):
        if symbol_source=='groundtruth':
            all_candidates=sum((groundtruth_candidates(page,cfg,run_id) for page in pages),[])
            candidates=[c for c in all_candidates if (c.page,c.line_id) in crops]
            omitted=len(all_candidates)-len(candidates)
            if omitted:
                record.notes.append('{} GT symbols omitted because their staff crops were dropped.'.format(
                    omitted))
            store.candidates={c.id:c for c in candidates}
            store.embeddings=_pool_candidate_embeddings(candidates,crops,extractor)
        else:
            all_candidates=[SymbolCandidate.from_dict(c.to_dict())
                            for c in source.ordered_candidates()]
            candidates=[c for c in all_candidates if (c.page,c.line_id) in crops]
            omitted=len(all_candidates)-len(candidates)
            if omitted:
                record.notes.append('{} discovered symbols omitted because their staff crops were dropped.'.format(
                    omitted))
            store.candidates={c.id:c for c in candidates}
            store.embeddings=np.array(source.embeddings,copy=True)
            gt=sum((groundtruth_symbols(page) for page in pages),[])
            spaces={(crop.page,crop.line_id):crop.staff_space_page for crop in crops.values()}
            note_gt=[g for g in gt if g.family=='note']
            _,matching=localization_metrics(candidates,note_gt,spaces,cfg.evaluation)
            matched=set(matching.candidate_to_gt)
            bulk_apply(store,[c.id for c in candidates if c.id in matched],review_state=ReviewState.ACCEPTED,
                       family=SymbolFamily.NOTE,scope='auto:minimal-review')
            for c in candidates:
                if c.id not in matched: reject(c,RejectionReason.ARTIFACT)
            record.notes.append('Discovered-symbol run used the documented GT-matching minimal-review surrogate.')
    method=build_grouping_method(cfg.grouping.method)
    with timer('group'):
        for crop in crops.values():
            line_symbols=[c for c in candidates if c.page==crop.page and c.line_id==crop.line_id and
                          c.review_state!=ReviewState.REJECTED]
            groups=method.group(crop,line_symbols,cfg.grouping)
            for component_ids in groups:
                components=[store.candidates[cid] for cid in component_ids]
                box=components[0].box
                for c in components[1:]: box=box.union(c.box)
                relations=method.relations(crop,line_symbols,component_ids,cfg.grouping)
                score=float(np.mean([r.evidence.get('gap_ink',0) for r in relations])) if relations else 1.0
                now=utc_now(); nid=uuid4().hex
                neume=NeumeCandidate(id=nid,grouping_run_id=run_id,symbols_run_id=source.run.run_id,
                                     book=cfg.book,page=crop.page,line_id=crop.line_id,box=box,
                                     component_ids=component_ids,relations=relations,grouping_score=score,
                                     created_at=now,updated_at=now)
                store.neumes[nid]=neume
                for cid in component_ids: store.candidates[cid].neume_id=nid
    with timer('embed_and_cluster'):
        store.neume_embeddings=embed_neumes(store,crops,extractor,cfg.neume_embedding)
        result=cluster_neumes(store,cfg.clustering)
        record.counts.update(candidates=len(candidates),neumes=len(store.neumes),
                             clusters=result.n_clusters,outliers=result.n_outliers)
    with timer('visualize'):
        for page in pages:
            neume_page_overlay(page,store,os.path.join(run_dir,NEUME_OVERLAY_DIR,page.page+'.jpg'))
        for cluster_id in sorted({n.cluster_id for n in store.neumes.values()}):
            neume_contact_sheet(store,cluster_id,crops,os.path.join(
                run_dir,NEUME_CLUSTER_DIR,'cluster_'+safe_cluster_id(cluster_id)+'.jpg'))
    _finish(store); run_evaluation(run_dir)
    return run_dir

def run_page_suggestions(cfg: RunConfig, out_root: str,
                         extractor_override: Optional[str] = None) -> str:
    """Rank uncorrected pages by embedding coverage for the next fine-tuning batch."""
    cfg = copy.deepcopy(cfg)
    if extractor_override:
        cfg.features.backend = extractor_override
    if cfg.page_suggestions.count < 1:
        raise ValueError('page suggestion count must be positive')

    seed_everything(cfg.seed)
    book = DatabaseBook(cfg.book)
    from database.book_index import prefill_page_progress
    all_pages = prefill_page_progress(book)
    by_name = {page.page: page for page in all_pages}

    requested = cfg.pages or sorted(by_name)
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError('pages do not exist in book {}: {}'.format(cfg.book, ', '.join(missing)))

    configured_corrected = cfg.page_suggestions.corrected_pages
    if configured_corrected is None:
        corrected_names = sorted(page.page for page in all_pages
                                 if page.page_progress().locked.get(Locks.SYMBOLS, False))
        corrected_source = 'symbols_lock'
    else:
        missing_corrected = sorted(set(configured_corrected) - set(by_name))
        if missing_corrected:
            raise ValueError('corrected pages do not exist in book {}: {}'.format(
                cfg.book, ', '.join(missing_corrected)))
        corrected_names = sorted(set(configured_corrected))
        corrected_source = 'config'

    candidate_names = sorted(set(requested) - set(corrected_names))
    embedded_names = sorted(set(candidate_names) | set(corrected_names))
    pages = [by_name[name] for name in embedded_names]
    run_id = _run_id(cfg)
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=False)
    record = _record(cfg, run_id, 'page_suggestions', pages)
    timer = StageTimer(record)
    extractor = build_feature_extractor(cfg.features)
    embeddings: Dict[str, np.ndarray] = {}
    stats = {}
    excluded = []

    with timer('load_features'):
        extractor.load()
        record.feature_extractor = extractor.describe()
        record.device = record.feature_extractor.get('device', '')
        for page in pages:
            expected_lines = len(page.pcgts().page.all_music_lines())
            record.n_lines_total += expected_lines
            if expected_lines == 0:
                stats[page.page] = PageEmbeddingStats(page.page, 0, 0, 0.0)
                excluded.append({'page': page.page, 'reason': 'no_music_lines'})
                continue
            crops = staff_crops_of_page(page, cfg.crop)
            record.n_lines_dropped += expected_lines - len(crops)
            embedding, page_stats = foreground_page_embedding(
                crops, extractor, cfg.page_suggestions, cfg.discovery)
            page_stats.page = page.page
            stats[page.page] = page_stats
            if embedding is None:
                reason = 'all_crops_dropped' if not crops else 'no_symbol_foreground'
                excluded.append({'page': page.page, 'reason': reason})
                continue
            embeddings[page.page] = embedding

    with timer('select'):
        suggestions = select_representative_pages(
            embeddings, candidate_names, corrected_names, cfg.page_suggestions.count)
        for suggestion in suggestions:
            page_stats = stats[suggestion.page]
            suggestion.n_music_lines = page_stats.n_music_lines
            suggestion.n_foreground_patches = page_stats.n_foreground_patches

    ordered_embedding_pages = sorted(embeddings)
    matrix = (np.stack([embeddings[name] for name in ordered_embedding_pages])
              if ordered_embedding_pages else np.zeros((0, 0), dtype=np.float32))
    np.save(os.path.join(run_dir, PAGE_EMBEDDINGS_FILE), matrix.astype(np.float32))
    write_text_atomic(os.path.join(run_dir, PAGE_EMBEDDING_INDEX_FILE), json.dumps({
        'schema_version': SCHEMA_VERSION,
        'pages': [
            {'embedding_index': index, **stats[name].to_dict()}
            for index, name in enumerate(ordered_embedding_pages)
        ],
    }, indent=2))

    payload = {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'book': cfg.book,
        'method': 'foreground_embedding_kcenter',
        'corrected_source': corrected_source,
        'corrected_pages': corrected_names,
        'embedded_corrected_pages': [name for name in corrected_names if name in embeddings],
        'candidate_pages': candidate_names,
        'suggestions': [suggestion.to_dict() for suggestion in suggestions],
        'excluded_pages': excluded,
    }
    write_text_atomic(os.path.join(run_dir, SUGGESTIONS_FILE), json.dumps(payload, indent=2))
    record.counts.update(
        corrected_pages=len(corrected_names),
        candidate_pages=len(candidate_names),
        embedded_pages=len(embeddings),
        suggestions=len(suggestions),
        excluded_pages=len(excluded),
    )
    record.finished_at = utc_now()
    record.peak_rss_bytes = peak_rss_bytes()
    record.peak_cuda_bytes = peak_cuda_bytes()
    run_payload = record.to_dict()
    run_payload['schema_version'] = SCHEMA_VERSION
    write_text_atomic(os.path.join(run_dir, RUN_FILE), json.dumps(run_payload, indent=2))
    return run_dir


def run_evaluation(run_dir: str) -> str:
    store=CandidateStore.load(run_dir); cfg=store.run.config
    _,pages=_pages(cfg)
    gt=sum((groundtruth_symbols(page) for page in pages),[])
    gt_neumes=sum((groundtruth_neumes(page) for page in pages),[])
    spaces={(page.page,line.id):line.avg_line_distance(
                default=page.pcgts().page.avg_staff_line_distance())
            for page in pages for line in page.pcgts().page.all_music_lines()}
    loc,matching=localization_metrics(store.ordered_candidates(),gt,spaces,cfg.evaluation)
    methods=sorted({c.discovery_method for c in store.candidates.values()
                    if c.discovery_method and c.discovery_method != 'groundtruth-derived-box'})
    if methods:
        loc['by_discovery_method']={}
        for method in methods:
            subset=[c for c in store.ordered_candidates() if c.discovery_method==method]
            loc['by_discovery_method'][method]=localization_metrics(
                subset,gt,spaces,cfg.evaluation)[0]
    matched_candidates=[c for c in store.ordered_candidates() if c.id in matching.candidate_to_gt]
    gt_by_id={g.id:g for g in gt}
    labels=[c.cluster_id for c in matched_candidates]
    families=[gt_by_id[matching.candidate_to_gt[c.id]].family for c in matched_candidates]
    subtypes=[gt_by_id[matching.candidate_to_gt[c.id]].subtype for c in matched_candidates]
    embeddings=np.stack([store.embedding_of(c) for c in matched_candidates]) if matched_candidates else np.zeros((0,1))
    if store.run.kind=='neumes' and store.run.symbol_source=='groundtruth':
        sym={'not_evaluated':'GT-derived symbols are input to grouping, not a symbol-clustering result.'}
    else:
        sym=cluster_metrics(labels,families,subtypes,embeddings,cfg.evaluation.retrieval_k)
    metrics={'localization':loc,'symbol_clustering':sym,'grouping':{},'neume_clustering':{},
             'annotation_efficiency':annotation_efficiency(store,sym)}
    if store.neumes:
        pred=[n.component_ids for n in store.ordered_neumes()]
        relations=sum((n.relations for n in store.neumes.values()),[])
        metrics['grouping']=grouping_metrics(pred,gt_neumes,matching,relations,gt)
        neumes=store.ordered_neumes(); nlabels=[n.cluster_id for n in neumes]
        size=[str(len(n.component_ids)) for n in neumes]
        pattern=['-'.join(r.kind for r in n.relations) or 'single' for n in neumes]
        raw=cluster_metrics(nlabels,size,pattern,store.neume_embeddings,cfg.evaluation.retrieval_k)
        raw['outlier_component_count_histogram']=raw.pop('outlier_family_histogram')
        metrics['neume_clustering']={
            'component_count':raw.pop('family'),'relation_pattern':raw.pop('subtype'),**raw}
    path=store.write_json(METRICS_FILE,metrics); write_report(run_dir,metrics,cfg)
    return path
