import json
import os
import shutil
import tempfile
import unittest

import ommr4all.settings as settings
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT']=os.path.join(BASE_DIR,'tests','storage')
settings.PRIVATE_MEDIA_ROOT=os.path.join(BASE_DIR,'tests','storage')
import django; django.setup()

from database import DatabaseBook
from omr.discovery.config import RunConfig
from omr.discovery.runner import run_evaluation, run_neume_grouping, run_symbol_discovery
from omr.discovery.store import CandidateStore


class DiscoveryPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book_name='discovery_pipeline_test'
        cls.book_path=os.path.join(settings.PRIVATE_MEDIA_ROOT,cls.book_name)
        if os.path.exists(cls.book_path): shutil.rmtree(cls.book_path)
        shutil.copytree(DatabaseBook('demo').local_path(),cls.book_path)
        cls.out=tempfile.mkdtemp(prefix='ommr-discovery-test-')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.book_path,ignore_errors=True); shutil.rmtree(cls.out,ignore_errors=True)

    def test_stub_pipeline_is_reproducible_and_groups_groundtruth(self):
        cfg=RunConfig(book=self.book_name,pages=['page_test_symbol_detection_001'],seed=0)
        cfg.features.backend='stub'; cfg.discovery.method='ink'
        cfg.clustering.method='kmeans'; cfg.clustering.n_clusters=5; cfg.clustering.seed=0
        first=run_symbol_discovery(cfg,self.out); second=run_symbol_discovery(cfg,self.out)
        a,b=CandidateStore.load(first),CandidateStore.load(second)
        self.assertGreaterEqual(len(a.candidates),50)
        self.assertEqual(len(a.candidates),len(b.candidates))
        signature=lambda st:[(c.page,c.line_id,round(c.box.x,8),round(c.box.y,8),c.cluster_id)
                             for c in st.ordered_candidates()]
        self.assertEqual(signature(a),signature(b))
        page=DatabaseBook(self.book_name).page('page_test_symbol_detection_001').pcgts().page
        lines={line.id:line for line in page.all_music_lines()}
        for c in a.candidates.values():
            line=lines[c.line_id]; ss=line.avg_line_distance(); bounds=line.aabb
            self.assertGreaterEqual(c.box.x,bounds.left()-ss); self.assertLessEqual(c.box.right(),bounds.right()+ss)
            self.assertGreaterEqual(c.box.y,bounds.top()-ss); self.assertLessEqual(c.box.bottom(),bounds.bottom()+ss)
        run=json.load(open(os.path.join(first,'run.json')))
        self.assertEqual(run['config']['seed'],0); self.assertEqual(run['seed'],0)
        self.assertTrue(run['stage_seconds'])
        self.assertTrue(os.path.getsize(os.path.join(first,'overlays','page_test_symbol_detection_001.jpg')))
        sheets=os.listdir(os.path.join(first,'clusters')); self.assertTrue(sheets)
        self.assertTrue(all(os.path.getsize(os.path.join(first,'clusters',x)) for x in sheets))

        neumes=run_neume_grouping(cfg,first,self.out,'groundtruth')
        metrics_path=run_evaluation(neumes); metrics=json.load(open(metrics_path))
        self.assertGreater(metrics['grouping']['pairwise_f1'],0.7)
        self.assertEqual(set(metrics),{'localization','symbol_clustering','grouping','neume_clustering',
                                      'annotation_efficiency'})
        self.assertTrue(os.path.exists(os.path.join(neumes,'neumes.json')))
        self.assertTrue(os.listdir(os.path.join(neumes,'neume_overlays')))
        self.assertTrue(os.listdir(os.path.join(neumes,'neume_clusters')))


if __name__=='__main__': unittest.main()
