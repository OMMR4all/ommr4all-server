"""Command-line entry point for the offline discovery prototypes.

    python -m omr.discovery.cli --help
"""
import argparse
import json
import os

from omr.discovery.config import RunConfig, deep_merge, load_config


def _parser():
    p=argparse.ArgumentParser(description='Offline embedding experiments for symbol annotation')
    sub=p.add_subparsers(dest='command',required=True)
    symbols=sub.add_parser('symbols',help='milestone 1: discover and cluster symbols')
    symbols.add_argument('--book'); symbols.add_argument('--pages',nargs='+')
    symbols.add_argument('--config'); symbols.add_argument('--out'); symbols.add_argument('--feature-extractor',choices=['dino','stub'])
    symbols.add_argument('--method',choices=['ink','tokencut','patch_peaks','ink+tokencut'])
    suggest=sub.add_parser('suggest-pages',
                           help='rank uncorrected pages for a diverse fine-tuning batch')
    suggest.add_argument('--book'); suggest.add_argument('--pages',nargs='+')
    suggest.add_argument('--count',type=int); suggest.add_argument('--corrected-pages',nargs='*')
    suggest.add_argument('--page-method', choices=['staff_foreground', 'whole_image'])
    suggest.add_argument('--config'); suggest.add_argument('--out')
    suggest.add_argument('--feature-extractor',choices=['dino','stub'])
    neumes=sub.add_parser('neumes',help='milestone 2: group and cluster neumes')
    neumes.add_argument('--symbols-run',required=True); neumes.add_argument('--symbol-source',choices=['discovered','groundtruth'],default='discovered')
    neumes.add_argument('--geometry-only',action='store_true'); neumes.add_argument('--config'); neumes.add_argument('--out')
    evaluate=sub.add_parser('evaluate',help='recompute metrics and report for a run')
    evaluate.add_argument('--run',required=True); evaluate.add_argument('--out')
    dump=sub.add_parser('dump-config',help='print or write the complete default configuration')
    dump.add_argument('--out')
    return p


def _bootstrap():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','ommr4all.settings')
    import django; django.setup()
    from django.conf import settings
    return os.environ.get('OMMR4ALL_DISCOVERY_OUT',os.path.join(str(settings.BASE_DIR),'storage_discovery'))


def _resolve(value,out_root):
    if os.path.isdir(value): return os.path.abspath(value)
    path=os.path.join(out_root,value)
    if not os.path.isdir(path): raise FileNotFoundError('run directory not found: '+value)
    return path


def _summary(run_dir):
    from omr.discovery.page_selection import SUGGESTIONS_FILE
    suggestions_path=os.path.join(run_dir,SUGGESTIONS_FILE)
    if os.path.exists(suggestions_path):
        with open(suggestions_path) as f: payload=json.load(f)
        print(run_dir)
        print('kind=page_suggestions corrected={} candidates={} suggestions={}'.format(
            len(payload['corrected_pages']),len(payload['candidate_pages']),
            len(payload['suggestions'])))
        for suggestion in payload['suggestions']:
            print('{rank}. {page} novelty={novelty_score:.4f}'.format(**suggestion))
        print('suggestions='+suggestions_path)
        return
    from omr.discovery.store import CandidateStore, METRICS_FILE
    store=CandidateStore.load(run_dir); metrics={}
    path=os.path.join(run_dir,METRICS_FILE)
    if os.path.exists(path):
        with open(path) as f: metrics=json.load(f)
    print(run_dir)
    print('kind={} candidates={} neumes={} lines={}/{} peak_rss={}'.format(
        store.run.kind,len(store.candidates),len(store.neumes),store.run.n_lines_total-store.run.n_lines_dropped,
        store.run.n_lines_total,store.run.peak_rss_bytes))
    if metrics.get('localization'): print('localization_f1={:.4f}'.format(metrics['localization'].get('f1',0)))
    if metrics.get('grouping'): print('pairwise_grouping_f1={:.4f}'.format(metrics['grouping'].get('pairwise_f1',0)))
    print('report='+os.path.join(run_dir,'report.md'))


def main(argv=None):
    args=_parser().parse_args(argv)
    if args.command=='dump-config':
        text=json.dumps(RunConfig().to_dict(),indent=2)
        if args.out:
            from database.file_write import write_text_atomic
            write_text_atomic(args.out,text+'\n'); print(args.out)
        else: print(text)
        return 0
    default_out=_bootstrap(); out_root=os.path.abspath(args.out or default_out)
    from omr.discovery.runner import (run_evaluation, run_neume_grouping, run_page_suggestions,
                                      run_symbol_discovery)
    if args.command=='symbols':
        overrides={}
        if args.book is not None: overrides['book']=args.book
        if args.pages is not None: overrides['pages']=args.pages
        if args.feature_extractor: overrides['features']={'backend':args.feature_extractor}
        if args.method: overrides['discovery']={'method':args.method}
        cfg=load_config(args.config,overrides)
        run_dir=run_symbol_discovery(cfg,out_root,args.feature_extractor)
    elif args.command=='suggest-pages':
        overrides={}
        if args.book is not None: overrides['book']=args.book
        if args.pages is not None: overrides['pages']=args.pages
        if args.feature_extractor: overrides['features']={'backend':args.feature_extractor}
        page_suggestions={}
        if args.count is not None: page_suggestions['count']=args.count
        if args.page_method is not None: page_suggestions['method']=args.page_method
        if args.corrected_pages is not None:
            page_suggestions['corrected_pages']=args.corrected_pages
        if page_suggestions: overrides['page_suggestions']=page_suggestions
        cfg=load_config(args.config,overrides)
        run_dir=run_page_suggestions(cfg,out_root,args.feature_extractor)
    elif args.command=='neumes':
        symbols_dir=_resolve(args.symbols_run,out_root)
        from omr.discovery.store import CandidateStore
        base=CandidateStore.load(symbols_dir).run.config.to_dict()
        if args.config:
            with open(args.config) as f: base=deep_merge(base,json.load(f))
        if args.geometry_only: base=deep_merge(base,{'grouping':{'geometry_only':True}})
        cfg=RunConfig.from_dict(base)
        run_dir=run_neume_grouping(cfg,symbols_dir,out_root,args.symbol_source)
    else:
        run_dir=_resolve(args.run,out_root); run_evaluation(run_dir)
    _summary(run_dir); return 0


if __name__=='__main__': raise SystemExit(main())
