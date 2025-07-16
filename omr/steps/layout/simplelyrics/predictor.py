import os.path

from database.model import Model, MetaId
from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.layout.drop_capitals.predictor import DropCapitalPredictor, LAYOUT_DROP_CAPITAL_MODEL_DEFAULT_NAME
from omr.steps.layout.predictor import LayoutAnalysisPredictor, PredictionType, PredictionResult, \
    PredictionCallback, AlgorithmPredictorSettings, FinalPredictionResult, IdCoordsPair
from typing import List, Optional
from database.file_formats.pcgts import PcGts, BlockType, Coords, Line, Rect, Point, Size
import numpy as np
from omr.steps.layout.lyricsbbs.meta import Meta
from omr.steps.step import Step
from loguru import logger


class Predictor(LayoutAnalysisPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        '''
        meta = Step.meta(AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL)
        from ommr4all.settings import BASE_DIR
        path = BASE_DIR + '/internal_storage/default_models/french14/layout_drop_capital/'
        model = Model(
            MetaId.from_custom_path(path,
                                    meta.type()))
        # print(model.path)
        settings = AlgorithmPredictorSettings(
            model=model,
        )

        if os.path.exists(os.path.join(settings.model.local_file(LAYOUT_DROP_CAPITAL_MODEL_DEFAULT_NAME))):
            self.drop_capital: DropCapitalPredictor = meta.create_predictor(settings)
        else:
            self.drop_capital: DropCapitalPredictor = None
        '''
        logger.info(self.settings.model.path)
        meta = Step.meta(AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO)

        #model = meta.default_model_for_style('french14')

        # print(model.path)
        settings = AlgorithmPredictorSettings(
            model=self.settings.model,
        )
        self.drop_capital: DropCapitalPredictor = meta.create_predictor(settings)

    def _predict_single(self, pcgts_file: PcGts) -> FinalPredictionResult:
        mls_in_cols = pcgts_file.page.all_music_lines_in_columns()
        music_coords: List[IdCoordsPair] = []
        lyric_coords: List[IdCoordsPair] = []
        for mls in mls_in_cols:
            mls = [ml for ml in mls if len(ml.staff_lines) > 0]
            mls.sort(key=lambda ml: ml.center_y())
            pad = pcgts_file.page.avg_staff_line_distance() / 2
            pad_tup = (0, pad)
            for ml in mls:
                ml.staff_lines.sort()

            for m in mls:
                music_coords.append(IdCoordsPair(Coords(np.concatenate(
                    [
                        m.staff_lines[0].coords.points - pad_tup,
                        [
                            c.coords.points[-1] for c in m.staff_lines[1:-1]
                        ],
                        m.staff_lines[-1].coords.points[::-1] + pad_tup,
                        [
                            c.coords.points[0] for c in reversed(m.staff_lines[1:-1])
                        ],
                    ]
                )), m.id))

            avg_staff_distance = np.mean(
                [m2.staff_lines[0].center_y() - m1.staff_lines[-1].center_y() for m1, m2 in zip(mls[:-1], mls[1:])])
            for m1, m2 in zip(mls[:-1], mls[1:]):
                top_l = m1.staff_lines[-1]
                bot_l = m2.staff_lines[0]
                bot_points = bot_l.coords.points[np.where((top_l.coords.points[0][0] < bot_l.coords.points[:, 0]) & (
                        bot_l.coords.points[:, 0] < top_l.coords.points[-1][0]))]
                bot_points = np.concatenate([
                    [(top_l.coords.points[0][0], bot_l.interpolate_y(top_l.coords.points[0][0]))],
                    bot_points,
                    [(top_l.coords.points[-1][0], bot_l.interpolate_y(top_l.coords.points[-1][0]))],
                ])
                lyric_coords.append(IdCoordsPair(Coords(np.concatenate([
                    top_l.coords.points + pad_tup,
                    [top_l.coords.points[-1] + (
                        0, bot_points[-1][1] - top_l.coords.interpolate_y(bot_points[-1][0])) - pad_tup],
                    bot_points[::-1] - pad_tup,
                    [top_l.coords.points[0] + (
                        0, bot_points[0][1] - top_l.coords.interpolate_y(bot_points[0][0])) - pad_tup],
                ], axis=0))))

            top_l = mls[-1].staff_lines[-1]
            lyric_coords.append(IdCoordsPair(Coords(np.concatenate((
                top_l.coords.points + pad_tup,
                top_l.coords.points[::-1] + (0, avg_staff_distance - pad),
            ), axis=0))))
        drop_capital_blocks = []
        # Todo filter drop capitals
        if self.settings.params.dropCapitals and self.drop_capital is not None:
            res = list(self.drop_capital._predict([pcgts_file]))[0]
            for x in res.blocks.get(BlockType.DROP_CAPITAL):
                drop_capital_blocks.append(IdCoordsPair(x))
            loop = True
            while loop:
                loop = False
                for x in range(len(drop_capital_blocks)):

                    for y in range(x + 1, len(drop_capital_blocks)):
                        rec1 = drop_capital_blocks[x].coords.aabb()
                        rec2 = drop_capital_blocks[y].coords.aabb()

                        if rec1.intersetcsWithRect(rec2):
                            smaller = x if rec1.area() < rec2.area() else y
                            del drop_capital_blocks[smaller]
                            loop = True
                            break

        # Todo improve drop capital-lyric matching
        if self.settings.params.documentStarts:
            w_dc = []
            for ind, i in enumerate(drop_capital_blocks):
                dc_rec = i.coords.aabb()
                dc_b = dc_rec.bottom()
                dc_t = dc_rec.top()
                dc_m = dc_rec.center
                lines = []
                for ind1, l in enumerate(lyric_coords):
                    if dc_b > l.coords.aabb().top() > dc_t:
                        lines.append(ind1)
                nearest = None
                min_distance = 99999999
                for ind2, l in enumerate(lines):
                    #min_d = min(i.coords.smallest_distance_between_polys(lyric_coords[l].coords))
                    if lyric_coords[l].coords.aabb().left() < i.coords.aabb().left() < lyric_coords[
                        l].coords.aabb().right() or abs(
                            i.coords.aabb().right() - lyric_coords[l].coords.aabb().left()) < 0.1:
                        point_1: Point = lyric_coords[l].coords.aabb().tl
                        point_2: Point = i.coords.aabb().tl
                        min_d = point_1.distance_sqr(point_2)
                        if min_d < min_distance:
                            nearest = l
                            min_distance = min_d
                if nearest is not None:
                    if lyric_coords[nearest].coords.aabb().left() < dc_rec.left():
                        # split lyric line
                        p1, p2 = lyric_coords[nearest].coords.split_polygon_by_x(dc_rec.left())

                        if p1.size > 0 and p2.size > 0:
                            p1 = Coords(p1)
                            p2 = Coords(p2)
                            if p2.aabb().left() <= p1.aabb().left():
                                t = p1
                                p1 = p2
                                p2 = t

                            lyric_coords[nearest] = IdCoordsPair(p1, None, lyric_coords[nearest].start)
                            lyric_coords.insert(nearest + 1, IdCoordsPair(p2, None, True))
                        else:
                            lyric_coords[nearest] = IdCoordsPair(p1, None, True)

                    else:
                        lyric_coords[nearest] = IdCoordsPair(lyric_coords[nearest].coords, lyric_coords[nearest].id,
                                                             True)
                else:
                    w_dc.append(ind)
            for i in reversed(w_dc):
                del (drop_capital_blocks[i])
            pass
        return FinalPredictionResult(
            blocks={
                BlockType.LYRICS: lyric_coords,
                BlockType.MUSIC: music_coords,
                BlockType.DROP_CAPITAL: drop_capital_blocks,

            },
            pcgts=pcgts_file,
        )

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        for i, r in enumerate(map(self._predict_single, pcgts_files)):
            if callback:
                callback.progress_updated(i / len(pcgts_files))
            yield r
