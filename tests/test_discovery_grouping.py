import unittest
from types import SimpleNamespace

import numpy as np

from database.file_formats.pcgts import Point
from omr.discovery.config import GroupingConfig, RunConfig
from omr.discovery.evaluation.grouping import grouping_metrics
from omr.discovery.evaluation.localization import Matching
from omr.discovery.grouping.rule_graph import RuleGraphNeumeGrouping
from omr.discovery.provenance import RunRecord
from omr.discovery.regions import StaffCrop
from omr.discovery.review import add_component, merge_groups, remove_component, split_group
from omr.discovery.schema import (Box, GroupingState, NeumeCandidate, ReviewState, SymbolCandidate,
                                  SymbolFamily, SymbolLabel, SymbolRelation)
from omr.discovery.store import CandidateStore


class IdentityPage:
    def page_to_image_scale(self,p,ref): return p
    def image_to_page_scale(self,p,ref): return p


class IdentityDataset:
    def global_to_local_pos(self,p,params): return p
    def local_to_global_pos(self,p,params): return p


def cand(cid,x):
    return SymbolCandidate(cid,'run','book','page','block','line',Box(x-3,47,6,6),x,50,
                           review_state=ReviewState.ACCEPTED,
                           label=SymbolLabel(SymbolFamily.NOTE,'normal'))


def crop():
    binary=np.zeros((100,180),bool)
    # A--B are visibly connected; C and D are geometrically close but separated by background.
    binary[47:54,27:44]=True
    binary[47:54,97:104]=True; binary[47:54,109:116]=True
    return StaffCrop('book','page','block','line',np.full((100,180,3),255,np.uint8),binary,
                     np.ones_like(binary),[],20.0,20.0,SimpleNamespace(id='line'),IdentityPage(),
                     IdentityDataset(),[],None)


class DiscoveryGroupingTest(unittest.TestCase):
    def setUp(self):
        self.a,self.b,self.c,self.d=cand('a',30),cand('b',40),cand('c',100),cand('d',112)
        self.symbols=[self.a,self.b,self.c,self.d]; self.crop=crop(); self.method=RuleGraphNeumeGrouping()

    def test_rule_graph_and_geometry_ablation(self):
        cfg=GroupingConfig(max_dx_staff_space=0.7,min_gap_ink=0.6)
        self.assertEqual(self.method.group(self.crop,self.symbols,cfg),[['a','b'],['c'],['d']])
        linked=self.method.relations(self.crop,self.symbols,['a','b'],cfg)
        separated=self.method.relations(self.crop,self.symbols,['c','d'],cfg)
        self.assertEqual(linked[0].kind,'looped'); self.assertEqual(separated[0].kind,'gaped')
        cfg.geometry_only=True
        self.assertEqual(self.method.group(self.crop,self.symbols,cfg),[['a','b'],['c','d']])

    def test_group_mutations_preserve_store_invariants(self):
        store=CandidateStore('/tmp/not-written',RunRecord(1,'run','neumes',RunConfig()))
        store.candidates={s.id:s for s in self.symbols}
        n=NeumeCandidate('n','g','run','book','page','line',self.a.box.union(self.b.box),['a','b'],
                         [SymbolRelation('a','b','looped')])
        store.neumes={'n':n}; self.a.neume_id=self.b.neume_id='n'
        add_component(store,n,'c'); self.assertEqual(n.grouping_state,GroupingState.MODIFIED)
        foreign=cand('foreign',150); foreign.page='other'; store.candidates[foreign.id]=foreign
        with self.assertRaises(ValueError): add_component(store,n,foreign.id)
        self.assertFalse(store.validate())
        remove_component(store,n,'c'); self.assertIsNone(self.c.neume_id); self.assertFalse(store.validate())
        left,right=split_group(store,n,1); self.assertFalse(store.validate())
        merged=merge_groups(store,[left.id,right.id]); self.assertEqual(merged.grouping_state,GroupingState.MODIFIED)
        self.assertFalse(store.validate())

    def test_grouping_metrics_perfect_merge_and_split(self):
        matching=Matching({x:x for x in 'abcd'},{x:x for x in 'abcd'},{})
        gt=[['a','b'],['c','d']]
        perfect=grouping_metrics(gt,gt,matching)
        self.assertEqual(perfect['pairwise_f1'],1.0); self.assertEqual(perfect['exact_group_match'],1.0)
        merged=grouping_metrics([['a','b','c','d']],gt,matching)
        self.assertEqual(merged['over_grouping_rate'],1.0); self.assertEqual(merged['under_grouping_rate'],0.0)
        split=grouping_metrics([['a'],['b'],['c','d']],gt,matching)
        self.assertEqual(split['over_grouping_rate'],0.0); self.assertEqual(split['under_grouping_rate'],0.5)
        self.assertEqual(split['exact_group_match'],0.5)


if __name__=='__main__': unittest.main()
