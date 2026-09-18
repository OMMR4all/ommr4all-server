"""Human-readable milestone report from measured data only."""
import json
import os
from collections import Counter

from database.file_write import write_text_atomic
from omr.discovery.store import CandidateStore, REPORT_FILE


def _table(mapping):
    return '\n'.join('| `{}` | `{}` |'.format(k, json.dumps(v, sort_keys=True)) for k,v in mapping.items())


def write_report(run_dir: str, metrics, cfg) -> None:
    store=CandidateStore.load(run_dir); run=store.run
    loc=metrics.get('localization',{}); sym=metrics.get('symbol_clustering',{})
    grouping=metrics.get('grouping',{}); neume=metrics.get('neume_clustering',{})
    counts=Counter(c.page for c in store.candidates.values())
    line_counts=Counter((c.page,c.line_id) for c in store.candidates.values())
    failures={
        'staff-line artifacts':sum(bool(c.rejection_reason and c.rejection_reason.value=='staff_line') for c in store.candidates.values()),
        'touching symbols / merges':loc.get('n_merges',0),
        'fragments / splits':loc.get('n_splits',0),
        'duplicate proposals':loc.get('n_duplicates',0),
        'missed small symbols':loc.get('missed_by_family',{}).get('note',0),
    }
    purity=(sym.get('family') or {}).get('weighted_purity')
    localisation_f1=loc.get('f1')
    grouping_f1=grouping.get('pairwise_f1')
    proceed=(localisation_f1 is None or localisation_f1>=0.5) and (grouping_f1 is None or grouping_f1>=0.7)
    recommendation='PROCEED to the next review gate' if proceed else 'STOP: inspect and improve the named failures before persistence/UI work'
    by_method=loc.get('by_discovery_method',{})
    if by_method:
        best_method=max(by_method,key=lambda name:by_method[name].get('f1',0.0))
        next_method=('Use `{}` as the next localisation baseline (F1 {:.3f}); keep the other '
                     'method experimental until its oversized/fragmented proposals improve.'
                     .format(best_method,by_method[best_method].get('f1',0.0)))
    else:
        next_method='No discovery-method comparison is available for this run.'
    surrogate='Yes — discovered-symbol grouping used a GT-matching minimal-review surrogate.' if run.symbol_source=='discovered' else 'No.'
    content=f"""# Symbol / Neume Discovery Run `{run.run_id}`

## Decision

**{recommendation}.** This is an offline experimental run. It did not modify PCGTS and is not a supervised-detector replacement.

**Next discovery-method recommendation:** {next_method}

## Reproducibility

- Kind: `{run.kind}`
- Parent run: `{run.parent_run_id}`
- Symbol source: `{run.symbol_source}`
- Seed: `{run.seed}`
- Pages: `{', '.join(run.pages)}`
- Feature extractor: `{json.dumps(run.feature_extractor, sort_keys=True)}`
- Git commit: `{run.environment.get('git_commit')}`
- Minimal-review surrogate: {surrogate}

## Counts

| Page | Candidates |
|---|---:|
{''.join('| `{}` | {} |\n'.format(k,v) for k,v in sorted(counts.items()))}

- Lines processed: {run.n_lines_total-run.n_lines_dropped}/{run.n_lines_total}
- Dropped lines: {run.n_lines_dropped}
- Neumes: {len(store.neumes)}

## Runtime and memory

| Stage | Seconds |
|---|---:|
{''.join('| `{}` | {:.3f} |\n'.format(k,v) for k,v in run.stage_seconds.items())}

- Peak RSS: {run.peak_rss_bytes} bytes
- Peak CUDA allocation: {run.peak_cuda_bytes}

## Localisation (graphical symbols)

PCGTS contains symbol centres, not boxes. Matching is Hungarian centre-distance matching in staff spaces. `iou_mean_derived_gt_box` uses an explicitly **derived pseudo-box**, not GT geometry.

| Metric | Value |
|---|---|
{_table(loc)}

## Symbol clustering (semantic-symbol review aid)

Cluster ids are similarity labels, never semantic classes. Family and subtype metrics are separate.

| Metric | Value |
|---|---|
{_table(sym)}

## Grouping into neumes

| Metric | Value |
|---|---|
{_table(grouping)}

## Neume clustering (semantic-neume review aid)

PCGTS does not contain `BasicNeumeType` ground truth on these fixtures. Cluster quality is
therefore measured against structural component-count and relation-pattern labels, not presented
as semantic neume-type accuracy.

| Metric | Value |
|---|---|
{_table(neume)}

## Annotation efficiency

This block contains interaction counts and a clearly labelled **simulation**, not timing claims.

| Metric | Value |
|---|---|
{_table(metrics.get('annotation_efficiency',{}))}

## Failure modes

| Failure mode | Count |
|---|---:|
{''.join('| {} | {} |\n'.format(k,v) for k,v in failures.items())}

Largest impure clusters: `{json.dumps(metrics.get('largest_impure_clusters', []))}`.

## Artifacts

- [Page overlays](overlays/)
- [Symbol cluster contact sheets](clusters/)
- [Neume overlays](neume_overlays/)
- [Neume cluster contact sheets](neume_clusters/)
- [Machine-readable metrics](metrics.json)

## Configuration

```json
{json.dumps(cfg.to_dict(), indent=2)}
```
"""
    write_text_atomic(os.path.join(run_dir, REPORT_FILE),content)
