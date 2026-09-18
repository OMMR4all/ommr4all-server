"""Pairwise and exact-set neume grouping evaluation."""
from collections import defaultdict
from itertools import combinations


def _pairs(groups):
    return {tuple(sorted(pair)) for group in groups for pair in combinations(group,2)}


def grouping_metrics(pred_groups, gt_groups, matching, relations=None, gt_symbols=None):
    # Compare in GT-id space; unmatched discovered components do not fabricate GT pairs.
    mapped=[]
    for group in pred_groups:
        ids=[matching.candidate_to_gt[cid] for cid in group if cid in matching.candidate_to_gt]
        if ids: mapped.append(ids)
    gt_lists=[list(getattr(g,'component_ids',g)) for g in gt_groups]
    pred_pairs, gt_pairs=_pairs(mapped),_pairs(gt_lists)
    tp=len(pred_pairs & gt_pairs); precision=tp/len(pred_pairs) if pred_pairs else (1.0 if not gt_pairs else 0.0)
    recall=tp/len(gt_pairs) if gt_pairs else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    pred_sets=[frozenset(g) for g in mapped]; gt_sets=[frozenset(g) for g in gt_lists]
    exact=sum(g in pred_sets for g in gt_sets)/len(gt_sets) if gt_sets else 1.0
    gt_owner={sid:i for i,g in enumerate(gt_lists) for sid in g}
    over=sum(len({gt_owner[s] for s in g if s in gt_owner})>=2 for g in mapped)/(len(mapped) or 1)
    pred_owner={sid:i for i,g in enumerate(mapped) for sid in g}
    under=sum(len({pred_owner[s] for s in g if s in pred_owner})>=2 for g in gt_lists)/(len(gt_lists) or 1)
    exact_orders=[g for g in mapped if frozenset(g) in gt_sets]
    order_ok=sum(g==gt_lists[gt_sets.index(frozenset(g))] for g in exact_orders)/(len(exact_orders) or 1)

    relation_hits=relation_total=0
    if relations is not None and gt_symbols is not None:
        gt_by_id={g.id:g for g in gt_symbols}
        for relation in relations:
            a=matching.candidate_to_gt.get(relation.from_id); b=matching.candidate_to_gt.get(relation.to_id)
            if not a or not b or b not in gt_by_id: continue
            expected='looped' if gt_by_id[b].connection=='looped' else 'gaped'
            relation_hits += relation.kind==expected; relation_total += 1
    by_size={}
    sizes=sorted({len(g) for g in gt_lists})
    for size in sizes:
        subset=[frozenset(g) for g in gt_lists if len(g)==size]
        by_size[str(size)]={'n':len(subset),'exact_group_match':sum(g in pred_sets for g in subset)/len(subset)}
    return {'pairwise_precision':precision,'pairwise_recall':recall,'pairwise_f1':f1,
            'exact_group_match':exact,'over_grouping_rate':over,'under_grouping_rate':under,
            'component_order_accuracy':order_ok,
            'relation_kind_accuracy':relation_hits/relation_total if relation_total else None,
            'relation_kind_n':relation_total,'by_component_count':by_size,
            'n_predicted_groups':len(pred_groups),'n_groundtruth_groups':len(gt_lists)}
