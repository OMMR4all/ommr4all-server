import json
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from omr.discovery.config import DiscoveryConfig, RunConfig
from database.file_write import write_text_atomic
from omr.discovery.discovery.base import DiscoveryContext, split_stacked
from omr.discovery.discovery.patch_peaks import PatchPeakDiscovery
from omr.discovery.features.base import SpatialFeatureMap
from omr.discovery.provenance import RunRecord
from omr.discovery.review import accept, bulk_apply, reject, set_label
from omr.discovery.schema import (Box, GroupingState, MaskRle, NeumeCandidate, RejectionReason,
                                  ReviewState, SymbolCandidate, SymbolFamily, SymbolLabel,
                                  SymbolRelation, validate_label)
from omr.discovery.store import CandidateStore, SchemaVersionMismatch


def candidate(cid='a', x=0.1, line='line'):
    return SymbolCandidate(cid,'run','book','page','block',line,Box(x,0.2,0.03,0.04),
                           x+0.015,0.22)


class DiscoveryCandidateTest(unittest.TestCase):
    def test_box_and_mask_roundtrip(self):
        a=Box(0,0,2,2); b=Box(1,1,2,2)
        self.assertAlmostEqual(a.iou(b),1/7); self.assertEqual(a.union(b),Box(0,0,3,3))
        mask=np.array([[0,1,1],[1,0,1]],bool)
        self.assertTrue(np.array_equal(mask,MaskRle.from_array(mask).to_array()))

    def test_stacked_noteheads_split_but_tall_single_symbol_does_not(self):
        cfg=DiscoveryConfig(); staff_space=20.0
        stacked=np.zeros((36,20),bool)
        stacked[3:15,4:16]=True; stacked[21:33,4:16]=True
        stacked[14:22,9:11]=True
        pieces=split_stacked(stacked,staff_space,cfg)
        self.assertEqual(len(pieces),2)
        self.assertTrue(all(piece.any() for piece,_ in pieces))

        single=np.zeros((36,20),bool)
        single[3:17,4:16]=True; single[16:33,9:11]=True
        self.assertEqual(split_stacked(single,staff_space,cfg),[(single,0)])
    def test_spatial_feature_map_padding_density_and_projection(self):
        feature_map=SpatialFeatureMap(np.ones((2,2,1),dtype=np.float32),patch_size=2,
                                      input_size=(4,4),content_size=(3,3),crop_size=(3,3))
        density=feature_map.area_density(np.ones((3,3),dtype=bool))
        np.testing.assert_allclose(density,np.array([[1.0,0.5],[0.5,0.25]]))

        border=np.array([[0,0],[0,1]],dtype=bool)
        projected=feature_map.grid_to_crop(border,nearest=True)>0
        self.assertEqual(projected.shape,(3,3))
        self.assertEqual(int(projected.sum()),1)
        self.assertTrue(projected[2,2])
        with self.assertRaises(ValueError):
            feature_map.area_density(np.ones((2,3),dtype=bool))
        with self.assertRaises(ValueError):
            feature_map.grid_to_crop(np.ones((2,3),dtype=float))
        with self.assertRaises(IndexError):
            feature_map.patch_center(2,0)
        inconsistent=feature_map._replace(input_size=(4,6))
        with self.assertRaises(ValueError):
            inconsistent.patch_of_pixel(0,0)

    def test_spatial_feature_map_uses_independent_axis_scales(self):
        features=np.array([[[1.0,0.0],[1.0,0.0]],
                           [[0.0,1.0],[0.0,1.0]]],dtype=np.float32)
        feature_map=SpatialFeatureMap(features,patch_size=2,input_size=(4,4),
                                      content_size=(3,4),crop_size=(6,2))

        self.assertEqual(feature_map.scale_xy,(2.0,0.5))
        self.assertEqual(feature_map.crop_pixels_per_patch(),(1.0,4.0))
        self.assertEqual(feature_map.patch_center(1,1),(1.5,5.0))
        self.assertEqual(feature_map.patch_of_pixel(1.25,4.5),(1,1))
        pooled=feature_map.pool_box(1.0,4.0,1.0,2.0)
        np.testing.assert_allclose(pooled,np.array([0.25,0.75])/np.linalg.norm([0.25,0.75]))

    def test_patch_peaks_local_background_retains_only_local_outlier(self):
        ink=np.zeros((40,80),dtype=bool)
        ink[10:20,50:60]=True
        ink[10:20,60:70]=True
        features=np.zeros((4,8,2),dtype=np.float32)
        features[:,:4,0]=1.0
        features[:,4:,1]=1.0
        features[1,6]=[0.0,-1.0]
        feature_map=SpatialFeatureMap(features,10,ink.shape,ink.shape,ink.shape)
        cfg=DiscoveryConfig(patch_peak_local_background_staff_space=1.0,
                            patch_peak_min_background_patches=4)
        crop=SimpleNamespace(region_mask=np.ones_like(ink),staff_space_px=10.0)

        proposals=PatchPeakDiscovery().discover(
            DiscoveryContext(crop,cfg,ink,feature_map))

        self.assertEqual(len(proposals),1)
        center=(proposals[0].x+proposals[0].w/2,proposals[0].y+proposals[0].h/2)
        np.testing.assert_allclose(center,(65.0,15.0),atol=1e-7)

    def test_patch_peaks_prominence_rejects_ridge_but_keeps_strict_peaks(self):
        ridge_ink=np.zeros((40,80),dtype=bool)
        ridge_ink[10:20,20:60]=True
        ridge_features=np.zeros((4,8,2),dtype=np.float32)
        ridge_features[...,0]=1.0
        ridge_features[1,2:6]=[-1.0,0.0]
        ridge_map=SpatialFeatureMap(ridge_features,10,ridge_ink.shape,
                                    ridge_ink.shape,ridge_ink.shape)
        cfg=DiscoveryConfig(patch_peak_local_background_staff_space=1.0,
                            patch_peak_min_background_patches=2,
                            patch_peak_prominence_staff_space=1.1)
        crop=SimpleNamespace(region_mask=np.ones_like(ridge_ink),staff_space_px=10.0)
        self.assertEqual(PatchPeakDiscovery().discover(
            DiscoveryContext(crop,cfg,ridge_ink,ridge_map)),[])

        ink=np.zeros((60,100),bool)
        ink[20:30,20:30]=True
        ink[20:30,70:80]=True
        ink[24:26,30:70]=True
        features=np.zeros((6,10,2),dtype=np.float32)
        features[...,0]=1.0
        features[2,2]=[-1.0,0.0]
        features[2,7]=[-1.0,0.0]
        feature_map=SpatialFeatureMap(features,10,ink.shape,ink.shape,ink.shape)
        crop=SimpleNamespace(region_mask=np.ones_like(ink),staff_space_px=20.0)
        ctx=DiscoveryContext(crop=crop,cfg=DiscoveryConfig(),ink=ink,feature_map=feature_map)

        first=PatchPeakDiscovery().discover(ctx)
        second=PatchPeakDiscovery().discover(ctx)

        self.assertEqual(len(first),2)
        self.assertEqual([proposal.method for proposal in first],['patch_peaks','patch_peaks'])
        self.assertTrue(all(proposal.mask is None for proposal in first))
        centers=[(proposal.x+proposal.w/2,proposal.y+proposal.h/2) for proposal in first]
        self.assertLessEqual(np.hypot(centers[0][0]-25,centers[0][1]-25),10)
        self.assertLessEqual(np.hypot(centers[1][0]-75,centers[1][1]-25),10)
        self.assertEqual([(p.x,p.y,p.w,p.h,p.score) for p in first],
                         [(p.x,p.y,p.w,p.h,p.score) for p in second])

    def test_patch_peaks_refines_center_and_falls_back_when_unweighted(self):
        ink=np.zeros((40,40),dtype=bool)
        ink[17:21,17:21]=True
        features=np.zeros((4,4,2),dtype=np.float32)
        features[...,0]=1.0
        features[1,1]=[-1.0,0.0]
        feature_map=SpatialFeatureMap(features,10,ink.shape,ink.shape,ink.shape)
        crop=SimpleNamespace(region_mask=np.ones_like(ink),staff_space_px=20.0)
        ctx=DiscoveryContext(crop,DiscoveryConfig(),ink,feature_map)

        proposal=PatchPeakDiscovery().discover(ctx)[0]
        refined=np.array([proposal.x+proposal.w/2,proposal.y+proposal.h/2])
        seed=np.array(feature_map.patch_center(1,1))
        centroid=np.array([19.0,19.0])
        self.assertLess(np.linalg.norm(refined-centroid),np.linalg.norm(seed-centroid))
        self.assertLessEqual(np.linalg.norm(refined-seed),10.0)

        with patch.object(SpatialFeatureMap,'grid_to_crop',
                          return_value=np.zeros(ink.shape,dtype=np.float32)):
            fallback=PatchPeakDiscovery().discover(ctx)[0]
        fallback_center=(fallback.x+fallback.w/2,fallback.y+fallback.h/2)
        self.assertEqual(fallback_center,tuple(seed))

    def test_patch_peaks_deterministic_sparse_and_zero_norm_fallbacks(self):
        ink=np.ones((20,30),dtype=bool)
        ink[:10,:10]=False
        features=np.zeros((2,3,2),dtype=np.float32)
        features[...,0]=1.0
        features[1,2]=[-1.0,0.0]
        feature_map=SpatialFeatureMap(features,10,ink.shape,ink.shape,ink.shape)
        cfg=DiscoveryConfig(patch_peak_min_background_patches=2)
        crop=SimpleNamespace(region_mask=np.ones_like(ink),staff_space_px=10.0)
        ctx=DiscoveryContext(crop,cfg,ink,feature_map)
        first=PatchPeakDiscovery().discover(ctx)
        second=PatchPeakDiscovery().discover(ctx)
        self.assertEqual(len(first),1)
        self.assertEqual([(p.x,p.y,p.w,p.h,p.score) for p in first],
                         [(p.x,p.y,p.w,p.h,p.score) for p in second])

        zero_features=np.array([[[1.0,0.0],[-1.0,0.0]],
                                [[1.0,0.0],[1.0,0.0]]],dtype=np.float32)
        zero_ink=np.ones((20,20),dtype=bool)
        zero_map=SpatialFeatureMap(zero_features,10,zero_ink.shape,
                                   zero_ink.shape,zero_ink.shape)
        self.assertEqual(PatchPeakDiscovery().discover(
            DiscoveryContext(SimpleNamespace(region_mask=np.ones_like(zero_ink),
                                             staff_space_px=10.0),
                             cfg,zero_ink,zero_map)),[])

    def test_tokencut_projection_excludes_fully_padded_tokens(self):
        feature_map=SpatialFeatureMap(np.ones((2,2,1),dtype=np.float32),patch_size=2,
                                      input_size=(4,4),content_size=(2,2),crop_size=(2,2))
        valid=feature_map.area_density(np.ones((2,2),dtype=bool))>=0.5
        np.testing.assert_array_equal(valid,np.array([[True,False],[False,False]]))
        padded_foreground=np.array([[False,False],[False,True]])
        self.assertFalse((feature_map.grid_to_crop(
            padded_foreground,nearest=True)>0).any())

    def test_patch_peaks_reject_unusable_inputs(self):
        ink=np.zeros((20,20),bool)
        features=np.zeros((2,2,2),dtype=np.float32)
        features[...,0]=1.0
        feature_map=SpatialFeatureMap(features,10,ink.shape,ink.shape,ink.shape)
        cfg=DiscoveryConfig()
        valid_crop=SimpleNamespace(region_mask=np.ones_like(ink),staff_space_px=10.0)
        invalid_crop=SimpleNamespace(region_mask=np.zeros_like(ink),staff_space_px=10.0)
        detector=PatchPeakDiscovery()

        self.assertEqual(detector.discover(
            DiscoveryContext(valid_crop,cfg,ink,feature_map)),[])
        nonempty=ink.copy()
        nonempty[5:15,5:15]=True
        self.assertEqual(detector.discover(
            DiscoveryContext(valid_crop,cfg,nonempty,None)),[])
        self.assertEqual(detector.discover(
            DiscoveryContext(invalid_crop,cfg,nonempty,feature_map)),[])

    def test_json_roundtrip_and_schema_version(self):
        c=candidate(); c.mask_rle=MaskRle.from_array(np.eye(3,dtype=bool)); c.manual_fields=['label.family']
        n=NeumeCandidate('n','g','run','book','page','line',c.box,[c.id],grouping_state=GroupingState.CONFIRMED)
        self.assertEqual(SymbolCandidate.from_json(c.to_json()),c)
        self.assertEqual(NeumeCandidate.from_json(n.to_json()),n)
        with tempfile.TemporaryDirectory() as root:
            run=RunRecord(1,'run','symbols',RunConfig())
            store=CandidateStore.create(root,run); store.candidates[c.id]=c; store.save()
            path=os.path.join(root,'candidates.json'); payload=json.load(open(path)); payload['schema_version']=999
            write_text_atomic(path,json.dumps(payload))
            with self.assertRaises(SchemaVersionMismatch): CandidateStore.load(root)

    def test_label_vocabulary_and_progression(self):
        self.assertFalse(validate_label(SymbolLabel(SymbolFamily.NOTE,'unknown')))
        self.assertFalse(validate_label(SymbolLabel(SymbolFamily.NOTE,'normal')))
        self.assertTrue(validate_label(SymbolLabel(SymbolFamily.NOT_A_SYMBOL,'normal')))
        self.assertTrue(validate_label(SymbolLabel(attributes={'made_up':'x'})))
        c=candidate(); set_label(c,family=SymbolFamily.NOTE); set_label(c,subtype='normal')
        set_label(c,attributes={'connection':'neume_start'})
        self.assertEqual(c.label,SymbolLabel(SymbolFamily.NOTE,'normal',{'connection':'neume_start'}))

    def test_review_states_and_background_evidence(self):
        c=candidate(); accept(c); self.assertEqual(c.review_state,ReviewState.ACCEPTED)
        self.assertEqual(c.label.family,SymbolFamily.UNKNOWN)
        reject(c,RejectionReason.ARTIFACT); self.assertFalse(c.rejection_reason.is_background_evidence())
        self.assertTrue(RejectionReason.STAFF_LINE.is_background_evidence())

    def test_bulk_respects_manual_fields_and_logs_once(self):
        run=RunRecord(1,'run','symbols',RunConfig()); store=CandidateStore('/tmp/not-written',run)
        a,b=candidate('a'),candidate('b',0.2); store.candidates={a.id:a,b.id:b}
        set_label(a,family=SymbolFamily.CLEF,manual=True)
        n=bulk_apply(store,['a','b'],review_state=ReviewState.ACCEPTED,family=SymbolFamily.NOTE,
                     attributes={'liquescent':'false'},scope='cluster:0')
        self.assertEqual(n,2); self.assertEqual(a.label.family,SymbolFamily.CLEF)
        self.assertEqual(a.label.attributes,{'liquescent':'false'})  # sibling field still updates
        self.assertEqual(b.label,SymbolLabel(SymbolFamily.NOTE,'unknown',{'liquescent':'false'}))
        self.assertEqual(len(store.interactions),1)

    def test_store_validation_catches_double_claim_and_broken_chain(self):
        run=RunRecord(1,'run','neumes',RunConfig()); store=CandidateStore('/tmp/not-written',run)
        a,b,c=candidate('a'),candidate('b',0.2),candidate('c',0.3); store.candidates={x.id:x for x in (a,b,c)}
        n1=NeumeCandidate('n1','g','run','book','page','line',a.box.union(b.box),['a','b'],
                          [SymbolRelation('a','b','gaped')])
        n2=NeumeCandidate('n2','g','run','book','page','line',b.box.union(c.box),['b','c'],
                          [SymbolRelation('b','c','gaped')])
        store.neumes={'n1':n1,'n2':n2}; a.neume_id='n1'; b.neume_id='n1'; c.neume_id='n2'
        errors='\n'.join(store.validate()); self.assertIn('claimed by',errors)
        n1.component_ids=['a','b','c']; n1.relations=[SymbolRelation('a','c','gaped')]
        self.assertIn('consecutive chain','\n'.join(store.validate()))


if __name__=='__main__': unittest.main()
