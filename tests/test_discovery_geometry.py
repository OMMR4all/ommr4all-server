import os
import unittest

import ommr4all.settings as settings
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT']=os.path.join(BASE_DIR,'tests','storage')
settings.PRIVATE_MEDIA_ROOT=os.path.join(BASE_DIR,'tests','storage')
import django; django.setup()

from database import DatabaseBook
from omr.discovery.config import CropConfig
from omr.discovery.regions import (crop_box_to_page, crop_point_to_page, page_point_to_crop,
                                   staff_crops_of_page)


class DiscoveryGeometryTest(unittest.TestCase):
    """Dewarping stays off because the reused operation's page->crop transform is not an inverse."""
    @classmethod
    def setUpClass(cls):
        cls.cfg=CropConfig(); cls.page=DatabaseBook('demo').page('page_test_symbol_detection_001')
        cls.crops=staff_crops_of_page(cls.page,cls.cfg)

    def test_dewarp_and_color_centering_are_disabled(self):
        self.assertFalse(self.cfg.dewarp)
        self.assertFalse(self.cfg.center)  # color `_resize_to_height` would stack 2-D padding onto RGB

    def test_symbol_centres_roundtrip(self):
        errors=[]
        for crop in self.crops:
            for symbol in crop.line.symbols:
                x,y=page_point_to_crop(crop,symbol.coord.x,symbol.coord.y)
                xx,yy=crop_point_to_page(crop,x,y)
                errors.append(max(abs(xx-symbol.coord.x),abs(yy-symbol.coord.y)))
        self.assertTrue(errors); self.assertLess(max(errors),1e-6)

    def test_box_size_matches_staff_space_scale(self):
        crop=self.crops[0]; side=0.8*crop.staff_space_px
        box=crop_box_to_page(crop,100,40,side,side)
        self.assertAlmostEqual(box.w/crop.staff_space_page,0.8,delta=0.02)
        self.assertAlmostEqual(box.h/crop.staff_space_page,0.8,delta=0.02)

    def test_staff_space_matches_mapped_staff_lines(self):
        for crop in self.crops:
            ys=[line[:,1].mean() for line in crop.staff_lines_px]
            distances=[b-a for a,b in zip(ys,ys[1:])]
            self.assertAlmostEqual(sum(distances)/len(distances),crop.staff_space_px,
                                   delta=0.1*crop.staff_space_px)
            # Without the broken colour-centering branch the staff region is about five spaces high.
            self.assertAlmostEqual(crop.staff_space_px,self.cfg.crop_height/5,delta=0.2*self.cfg.crop_height/5)


if __name__=='__main__': unittest.main()
