import os
import random

import shapely.geometry
from PIL import Image
from shapely.geometry import Polygon

from database.file_formats.pcgts.page import SymbolErrorType
from omr.confidence.symbol_sequence_confidence import SymbolSequenceConfidenceLookUp, SequenceSetting

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import List, Optional, Generator
from ocr4all_pixel_classifier.lib.predictor import Predictor, PredictSettings
from database.file_formats.pcgts import *
from omr.steps.symboldetection.dataset import SymbolDetectionDataset
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings
import cv2
import numpy as np
from omr.steps.symboldetection.pixelclassifier.meta import Meta
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.symboldetection.predictor import SymbolsPredictor, SingleLinePredictionResult
from omr.steps.symboldetection.post_processing.initial_based_correction import correct_symbols_inside_wrong_blocks, \
    correct_symbols_inside_text_blocks


def render_prediction_labels(labels, img=None):
    from shared.pcgtscanvas import PcGtsCanvas
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    if img is not None:
        out = np.stack((img,) * 3, axis=-1).astype(int)

    def draw(i, c):
        return np.kron((labels == i), c).reshape(out.shape).astype(int)

    for i, c in [
        (SymbolLabel.BACKGROUND, (255, 255, 255)),
        (SymbolLabel.NOTE_START, (255, 0, 0)),
        (SymbolLabel.NOTE_GAPPED, (255, 120, 120)),
        (SymbolLabel.NOTE_LOOPED, (120, 0, 0)),
        (SymbolLabel.CLEF_C, (120, 255, 120)),
        (SymbolLabel.CLEF_F, (0, 255, 0)),
        (SymbolLabel.ACCID_NATURAL, (0, 0, 255)),
        (SymbolLabel.ACCID_SHARP, (50, 50, 255)),
        (SymbolLabel.ACCID_FLAT, (0, 0, 120)),
    ]:
        c = PcGtsCanvas.color_for_music_symbol(i.to_music_symbol(), inverted=True, default_color=(255, 255, 255))
        if c != (0, 0, 0):
            out[:, :, 0] = np.where(labels == i, c[0], out[:, :, 0])
            out[:, :, 1] = np.where(labels == i, c[1], out[:, :, 1])
            out[:, :, 2] = np.where(labels == i, c[2], out[:, :, 2])

    # if img is not None:
    # out = (out.astype(float) * np.stack((img,) * 3, axis=-1) / 255).astype(np.uint8)

    return out.clip(0, 255).astype(np.uint8)


def filter_unique_symbols_by_coord(symbol_list1: List[MusicSymbol], symbol_list2: List[MusicSymbol],
                                   max_distance=0.00001):
    symbol_list = []
    for x in symbol_list2:
        for y in symbol_list1:
            if x.coord.distance_sqr(y.coord) < max_distance:
                break
        else:
            x.symbol_confidence.symbol_error_type = SymbolErrorType.SEGMENTATION
            symbol_list.append(x)
    return symbol_list


def fix_overlapping_symbols(page, symbols: List[MusicSymbol], scale_reference, debug=False):
    delete = []
    if debug:
        image2 = np.array(Image.open(page.location.file(scale_reference.file("color")).local_path()))

    def scale(x):
        return np.round(page.page_to_image_scale(x, scale_reference)).astype(int)

    avg_line_distance = page.avg_staff_line_distance()
    # print(avg_line_distance)
    # print(scale(avg_line_distance))
    avg_line_distance = scale(avg_line_distance)

    ##print(avg_line_distance)
    # self.img[int(pos[1] - self.avg_line_distance * 0.8):int(pos[1] + self.avg_line_distance * 0.8),
    # int(pos[0] - self.avg_line_distance * 0.3):int(pos[0] + self.avg_line_distance * 0.3)] = color

    def rect(symbol: MusicSymbol):
        coord_symbol = scale(symbol.coord)
        if symbol.symbol_type == symbol.symbol_type.CLEF:
            if symbol.clef_type == symbol.clef_type.C:
                return shapely.geometry.box(coord_symbol.x - avg_line_distance * 0.3,
                                            coord_symbol.y - avg_line_distance * 0.8,
                                            coord_symbol.x + avg_line_distance * 0.3,
                                            coord_symbol.y + avg_line_distance * 0.8)

            else:
                return shapely.geometry.box(coord_symbol.x - avg_line_distance * 0.4,
                                            coord_symbol.y - avg_line_distance * 0.8,
                                            coord_symbol.x + avg_line_distance * 0.4,
                                            coord_symbol.y + avg_line_distance * 0.8)

        else:

            return shapely.geometry.box(coord_symbol.x - avg_line_distance / 4 * 0.8,
                                        coord_symbol.y - avg_line_distance / 4 * 0.8,
                                        coord_symbol.x + avg_line_distance / 4 * 0.8,
                                        coord_symbol.y + avg_line_distance / 4 * 0.8)

    for symbol_ind in range(len(symbols)):
        symbol = symbols[symbol_ind]
        rect1 = rect(symbol)
        # print(rect1.bounds)
        # print(int(rect1.bounds[0]))
        # print(int(random.random() * 255), int(random.random() * 255) ,int(random.random() * 255))
        if debug:
            image2[int(rect1.bounds[1]): int(rect1.bounds[3]), int(rect1.bounds[0]): int(rect1.bounds[2])] = int(
                random.random() * 255), int(random.random() * 255), int(random.random() * 255)
        for symbol_ind2 in range(len(symbols) - (symbol_ind + 1)):
            symbol2 = symbols[symbol_ind2 + (symbol_ind + 1)]
            rect2 = rect(symbol2)

            # polygon = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
            # other_polygon = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])

            intersection = rect1.intersection(rect2)
            if intersection.area > 0:
                # print("new")
                # print(page.location.page)
                # print(symbol.coord)
                # print(symbol2.coord)
                # print(intersection)
                # print(intersection.area)
                if symbol.symbol_type is symbol.symbol_type.CLEF or symbol2.symbol_type is symbol.symbol_type.CLEF:
                    if symbol.symbol_type is symbol.symbol_type.CLEF:
                        delete.append(symbol_ind2 + (symbol_ind + 1))
                        # print(symbol_ind2 + (symbol_ind + 1))
                    else:
                        delete.append(symbol_ind)
                        # print(symbol_ind)
    if debug:
        from matplotlib import pyplot as plt
        plt.imshow(image2)
        plt.show()
    for i in reversed(delete):
        # print("Overlapping clef")
        del symbols[i]
    return symbols


def fix_missing_clef(symbols1: List[MusicSymbol], symbols2: List[MusicSymbol]):
    if len(symbols1) > 0:
        if symbols1[0].symbol_type != symbols1[0].symbol_type.CLEF:
            if len(symbols2) > 0:
                if symbols2[0].symbol_type == symbols2[0].symbol_type.CLEF:
                    symbol = symbols2[0]
                    symbol.symbol_confidence = None
                    symbols1.insert(0, symbol)
                    del symbols2[0]
    return symbols1, True


def fix_missing_clef2(symbols1: List[MusicSymbol], symbols2: List[MusicSymbol], page: Page, m: RegionLineMaskData):
    avg_line_distance = page.avg_staff_line_distance()

    def get_firstsymbols(symbols: List[MusicSymbol]):
        symbol = [symbols[0]]
        for symbol_next in symbols[1:]:
            if symbol_next.coord.x - symbol[0].coord.x < avg_line_distance / 5:
                symbol.append(symbol_next)
        return symbol

    m.operation.music_line.staff_lines.max_x_start()
    if len(symbols1) > 0:
        if symbols1[0].symbol_type != symbols1[0].symbol_type.CLEF:
            symbols3 = symbols1 + symbols2
            symbols3.sort(key=lambda key: key.coord.x)
            symbol = get_firstsymbols(symbols3)

            def average(list):
                return sum(list) / len(list)

            average_x = average([s.coord.x for s in symbol])
            average_y = average([s.coord.y for s in symbol])
            coord = Point(average_x, y=average_y)
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord, clef=True)
            # coord_updated = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(coord.x,
            #                                                                                      position_in_staff)
            if m.operation.music_line.staff_lines.max_x_start() + avg_line_distance * 3 / 4 > coord.x:
                symbols1.insert(0, create_clef(ClefType.C, coord=coord, position_in_staff=position_in_staff,
                                               confidence=None))
            else:

                pass

    return symbols1


def fix_pos_of_close_symbols(page, symbols: List[MusicSymbol], scale_reference, debug=False, m=None):
    if len(symbols) > 0:
        avg_line_distance = page.avg_staff_line_distance()

        def distance_bet_symbols(symbol1: MusicSymbol, symbol2: MusicSymbol):
            distance = symbol2.coord.x - symbol1.coord.x
            return distance

        def closer_to_staff_line(symbol1, symbol2, line):
            distance1 = abs(line.interpolate_y(symbol1.coord.x) - symbol1.coord.x)
            distance2 = abs(line.interpolate_y(symbol2.coord.x) - symbol2.coord.x)
            return symbol2 if distance1 < distance2 else symbol1
            pass

        def nearest_staffline(symbol1, symbol2, stafflines: StaffLines):
            distance = 9999999
            nearest_line = None
            for line in stafflines.sorted():
                distance_symbols = line.interpolate_y(symbol1.coord.x) + line.interpolate_y(symbol2.coord.x)
                if distance_symbols < distance:
                    distance = distance_symbols
                    nearest_line = line
            return nearest_line

        prev_symbol: MusicSymbol = symbols[0]
        for ind, symbol in enumerate(symbols[1:]):
            distance = distance_bet_symbols(prev_symbol, symbol)

            if distance < avg_line_distance / 5:
                pos1 = prev_symbol.position_in_staff
                pos2 = symbol.position_in_staff
                line = nearest_staffline(prev_symbol, symbol, m.operation.music_line.staff_lines)
                symbol_closer_to_staff_line = closer_to_staff_line(prev_symbol, symbol, line)
                if abs(pos1.value - pos2.value) == 2:
                    # check if error
                    coord_original = symbol_closer_to_staff_line.coord

                    coord = symbol_closer_to_staff_line.coord
                    pos = symbol_closer_to_staff_line.position_in_staff
                    #value =
                    while True:
                        coord = Point(x=coord.x, y=coord.y + 0.001)
                        new_pos = m.operation.music_line.compute_position_in_staff(coord)
                        if pos != new_pos:
                            print("found")
                            symbol_closer_to_staff_line.coord = coord
                            print(coord.x - coord_original.x)
                            break
            prev_symbol = symbol
    return symbols

def fix_pos_of_close_symbols2(page, symbols: List[MusicSymbol], scale_reference, debug=False, m=None):
    if len(symbols) > 0:
        avg_line_distance = page.avg_staff_line_distance()
        avg_line_distance_step = avg_line_distance / 50


        def distance_bet_symbols(symbol1: MusicSymbol, symbol2: MusicSymbol):
            distance = symbol2.coord.x - symbol1.coord.x
            return distance

        def closer_to_staff_line(symbol1, symbol2, line):
            distance1 = abs(line.interpolate_y(symbol1.coord.x) - symbol1.coord.x)
            distance2 = abs(line.interpolate_y(symbol2.coord.x) - symbol2.coord.x)
            return symbol2 if distance1 < distance2 else symbol1
            pass

        def nearest_staffline(symbol1, symbol2, stafflines: StaffLines):
            distance = 9999999
            nearest_line = None
            step = 0.1
            for line in stafflines.sorted():
                distance_symbols = line.interpolate_y(symbol1.coord.x) + line.interpolate_y(symbol2.coord.x)
                if distance_symbols < distance:
                    distance = distance_symbols
                    nearest_line = line
            return nearest_line
        def get_previous_symbols_with_same_pis(symbols: List[MusicSymbol], pis: MusicSymbolPositionInStaff):
            symbols_new=[]
            for x in symbols:
                if x.position_in_staff == pis:
                    symbols_new.append(x)
            return symbols_new

        #print("newline")
        for ind, symbol in enumerate(symbols):
            snapped_pos = m.operation.music_line.staff_lines.snap_to_pos(symbol.coord)

            coord = symbol.coord
            pis = symbol.position_in_staff
            its = 0
            while True:
                if snapped_pos > symbol.coord.y:
                    coord = Point(x=coord.x, y=coord.y - avg_line_distance_step)
                else:
                    coord = Point(x=coord.x, y=coord.y + avg_line_distance_step)
                #print(coord.y)
                new_pis = m.operation.music_line.compute_position_in_staff(coord)
                if pis != new_pis:
                    if its == 0:
                        #symbol.note_type = symbol.note_type.APOSTROPHA
                        prev_symbol = None
                        next_symbol = None
                        distance_1 = None
                        distance_2 = None
                        if ind > 1:
                            prev_symbol = symbols[ind -1]
                        if ind < len(symbols) -1:
                            next_symbol = symbols[ind +1]
                        if prev_symbol:
                            distance_1 = distance_bet_symbols(prev_symbol, symbol)
                        if next_symbol:
                            distance_2 = distance_bet_symbols(symbol, next_symbol)
                        symbol_to_compare = None
                        if distance_1 and distance_1 < avg_line_distance / 2:
                            symbol_to_compare = prev_symbol
                        elif distance_2 and distance_2 < avg_line_distance / 2:
                            symbol_to_compare = next_symbol
                        if symbol_to_compare and abs(symbol_to_compare.position_in_staff.value - new_pis.value) == 1:
                            #symbol.note_type = symbol.note_type.ORISCUS
                            symbol.coord = coord
                            symbol.position_in_staff = new_pis

                        #symbol.coord= coord
                    #print(its)

                    break
                if its > 1:
                    break
                its +=1

    return symbols

def correct_looped_connection(symbols1: List[MusicSymbol], symbols2: List[MusicSymbol], page: Page,
                              m: RegionLineMaskData):
    if len(symbols1) > 0:
        avg_line_distance = page.avg_staff_line_distance()
        prev_symbol: MusicSymbol = symbols1[0]

        def distance_bet_symbols(symbol1: MusicSymbol, symbol2: MusicSymbol):
            distance = symbol2.coord.x - symbol1.coord.x
            return distance

        def get_symbols_between(x1, x2, symbols_22: List[MusicSymbol]):
            filtered_symbols = []
            for ind, i in enumerate(symbols_22):
                if x1 < i.coord.x < x2:
                    filtered_symbols.append((i, ind))
                elif i.coord.x > x2:
                    break
            return filtered_symbols

        insert_symbols = []
        for ind, symbol in enumerate(symbols1[1:]):
            distance = distance_bet_symbols(prev_symbol, symbol)
            if symbol.symbol_type == symbol.symbol_type.NOTE \
                    and symbol.graphical_connection == symbol.graphical_connection.LOOPED:
                if distance > avg_line_distance:
                    symbols_between = get_symbols_between(prev_symbol.coord.x, symbol.coord.x, symbols2)
                    if len(symbols_between) > 0:
                        if len(symbols_between) == 1:
                            if symbols_between[0][0].position_in_staff != symbol.position_in_staff and \
                                    symbols_between[0][0].graphical_connection == symbols_between[0][0].graphical_connection.NEUME_START:

                                if 3 < symbols_between[0][0].position_in_staff < 10:
                                    insert_symbols.append((symbols_between[0][0], ind + 1, symbols_between[0][1]))
                                    # print(symbol.id)
                                    # print(symbol.graphical_connection)
                                    # print(m.operation.music_line.id)
                                    # print(prev_symbol.position_in_staff)
                                    # print("looped")
                                    # print(page.location.page)
                    else:
                        symbol.graphical_connection = symbol.graphical_connection.NEUME_START
                        # print(symbol.id)
                        # print(symbol.graphical_connection)
                        # print(m.operation.music_line.id)

                    # print("looped")
                    # print(page.location.page)
            else:
                if distance < avg_line_distance / 3:
                    # print("prob looped")
                    # print(page.location.page)
                    symbol.graphical_connection = symbol.graphical_connection.LOOPED

            prev_symbol = symbol
        for x, s_ind, sa_ind in reversed(insert_symbols):
            # x.note_type = x.note_type.ORISCUS
            del symbols2[sa_ind]
            symbols1.insert(s_ind, x)


class PCPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        settings = PredictSettings(
            n_classes=len(SymbolLabel),
            network=os.path.join(settings.model.local_file('model.h5'))
        )
        self.predictor = Predictor(settings)
        self.look_up = SymbolSequenceConfidenceLookUp(SequenceSetting.NOTE_3GRAM)

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDataset(pcgts_files, self.dataset_params)
        clefs = []
        for p in self.predictor.predict(dataset.to_page_segmentation_dataset()):
            m: RegionLineMaskData = p.data.user_data
            # from PIL import Image
            # rr = np.random.randint(0, 999999)
            # mask = m.mask
            # mask = np.stack([mask, mask, mask], axis=-1)
            # mask[mask == 1] = 255
            # print(mask.shape)
            # print(mask.dtype)
            # mask = np.where(mask[..., [0]] == 1, (255, 0, 0), mask)
            # mask = np.where(mask[..., [0]] == 2, (255,255, 0), mask)
            # mask = np.where(mask[..., [0]] == 3, (255, 0, 255), mask)
            # mask = np.where(mask[..., [0]] == 4, (0,255, 0), mask)
            # mask = np.where(mask[..., [0]] == 5, (0, 0, 255), mask)
            # mask = np.where(mask[..., [0]] == 6, (0,255, 255), mask)
            # mask = np.where(mask[..., [0]] == 7, (0, 125, 255), mask)
            # mask = mask.astype("uint8")
            # print(mask.shape)
            # print(mask.dtype)

            # img1 = Image.fromarray(mask)

            # img2 = Image.fromarray(m.line_image)
            # img1.save(os.path.join("/tmp", str(rr) + "mask.png"))
            # img2.save(os.path.join("/tmp", str(rr) + "image.png"))

            symbols = self.extract_symbols(p.probabilities, p.labels, m, dataset,
                                           clef=self.settings.params.use_clef_pos_correction)
            additional_symbols = filter_unique_symbols_by_coord(symbols,
                                                                self.extract_symbols(p.probabilities, p.labels, m,
                                                                                     dataset,
                                                                                     probability=0.95,
                                                                                     clef=self.settings.params.use_clef_pos_correction))
            if True:
                symbols = correct_symbols_inside_wrong_blocks(m.operation.page, symbols)
                symbols = correct_symbols_inside_text_blocks(m.operation.page, symbols)
                symbols = fix_overlapping_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2)

                additional_symbols = correct_symbols_inside_text_blocks(m.operation.page, additional_symbols)
                additional_symbols = correct_symbols_inside_wrong_blocks(m.operation.page, additional_symbols)
                additional_symbols = correct_symbols_inside_text_blocks(m.operation.page, additional_symbols)

                symbols, change = fix_missing_clef(symbols, additional_symbols)
                symbols = fix_missing_clef2(symbols1=symbols, symbols2=additional_symbols, page=m.operation.page, m=m)
                symbols = fix_overlapping_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2)
                #symbols = fix_pos_of_close_symbols(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m)
                correct_looped_connection(symbols, additional_symbols, page=m.operation.page, m=m)
                symbols = fix_pos_of_close_symbols2(m.operation.page, symbols, PageScaleReference.NORMALIZED_X2, m=m)

                initial_clef = None
                if len(symbols) > 0:
                    if symbols[0].symbol_type == symbols[0].symbol_type.CLEF:
                        clefs.append(symbols[0])
                        initial_clef = symbols[0]
                    elif len(clefs) > 0:
                        # symbols.insert(0, clefs[-1])
                        initial_clef = clefs[-1]
                line = Line(symbols=symbols)
                line.update_note_names(initial_clef=initial_clef)
                symbols = line.symbols
                '''
                if len(symbols) > 0:
                    if symbols[0].symbol_type != symbols[0].symbol_type.CLEF:
                        print(symbols[0].symbol_type)
                        if len(additional_symbols) > 0:
                            print(additional_symbols[0].symbol_type)
                        print(m.operation.page.location.page)
                '''
            single_line_symbols = SingleLinePredictionResult(symbols,
                                                             p.data.user_data)
            single_line_symbols_2 = SingleLinePredictionResult(additional_symbols,
                                                               p.data.user_data)

            if False:
                from shared.pcgtscanvas import PcGtsCanvas
                canvas = PcGtsCanvas(m.operation.page, PageScaleReference.NORMALIZED_X2)
                # for s in single_line_symbols.symbols:
                #    s.coord = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(s.coord.x,
                #                                                                                    s.position_in_staff)
                canvas.draw(single_line_symbols.symbols, invert=True)
                canvas.show()
            if False:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots(5, 1, sharey='all', sharex='all')
                ax[0].imshow(p.probabilities[:, :, 0], vmin=0.0, vmax=1.0)
                ax[1].imshow(p.data.image, vmin=0.0, vmax=255)
                ax[2].imshow(render_prediction_labels(p.labels, p.data.image))
                ax[3].imshow((p.probabilities[:, :, 0] <= 0.8) * (1 + np.argmax(p.probabilities[:, :, 1:], axis=-1)))
                ax[4].imshow(render_prediction_labels(p.data.mask, p.data.image))
                plt.show()

            yield single_line_symbols, single_line_symbols_2

    def extract_symbols(self, probs: np.ndarray, p: np.ndarray, m: RegionLineMaskData,
                        dataset: SymbolDetectionDataset, probability=0.5, clef=True) -> List[MusicSymbol]:
        # n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(((probs[:, :, 0] < 0.5) | (p > 0)).astype(np.uint8))
        p = (np.argmax(probs[:, :, 1:], axis=-1) + 1) * (probs[:, :, 0] < probability)
        n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8))
        symbols = []
        sorted_labels = sorted(range(1, n_labels), key=lambda i: (centroids[i, 0], -centroids[i, 1]))
        centroids_canvas = np.zeros(p.shape, dtype=np.uint8)
        for i in sorted_labels:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            if a <= 4:
                continue
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]
            c = Point(x=centroids[i, 0], y=centroids[i, 1])
            coord = dataset.local_to_global_pos(c, m.operation.params)
            coord = m.operation.page.image_to_page_scale(coord, m.operation.scale_reference)
            # coord = coord.round().astype(int)

            # compute label this the label with the highest frequency of the connected component
            area = p[y:y + h, x:x + w] * (cc[y:y + h, x:x + w] == i)

            label = SymbolLabel(int(np.argmax([np.sum(area == v + 1) for v in range(len(SymbolLabel) - 1)])) + 1)
            centroids_canvas[int(np.round(c.y)), int(np.round(c.x))] = label

            # confidences
            indexes_of_cc = np.where(cc == i)
            labels_of_cc = p[indexes_of_cc]
            probs_of_cc = probs[indexes_of_cc]
            avg_prob_cc = np.mean(probs_of_cc, axis=0)
            symbol_pred = SymbolPredictionConfidence(*avg_prob_cc.tolist())
            if label == SymbolLabel.NOTE_START:
                position_in_staff = m.operation.music_line.compute_position_in_staff(coord)
                # confidence_position_in_staff = m.operation.music_line.compute_confidence_position_in_staff(coord)
                # print(confidence_position_in_staff)
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.NEUME_START,
                    confidence=SymbolConfidence(symbol_pred, None)
                ))
            elif label == SymbolLabel.NOTE_GAPPED:
                position_in_staff = m.operation.music_line.compute_position_in_staff(coord)

                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.GAPED,
                    confidence=SymbolConfidence(symbol_pred, None)

                ))
            elif label == SymbolLabel.NOTE_LOOPED:
                position_in_staff = m.operation.music_line.compute_position_in_staff(coord)

                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.LOOPED,
                    confidence=SymbolConfidence(symbol_pred, None)

                ))
            elif label == SymbolLabel.CLEF_C:
                position_in_staff = m.operation.music_line.compute_position_in_staff(coord, clef=clef)
                coord_updated = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(coord.x,
                                                                                                      position_in_staff)
                symbols.append(create_clef(ClefType.C, coord=coord_updated, position_in_staff=position_in_staff,
                                           confidence=SymbolConfidence(symbol_pred, None)))
            elif label == SymbolLabel.CLEF_F:
                position_in_staff = m.operation.music_line.compute_position_in_staff(coord, clef=clef)
                coord_updated = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(coord.x,
                                                                                                      position_in_staff)
                symbols.append(create_clef(ClefType.F, coord=coord_updated, position_in_staff=position_in_staff,
                                           confidence=SymbolConfidence(symbol_pred, None)))
            elif label == SymbolLabel.ACCID_FLAT:
                symbols.append(
                    create_accid(AccidType.FLAT, coord=coord, confidence=SymbolConfidence(symbol_pred, None)))
            elif label == SymbolLabel.ACCID_SHARP:
                symbols.append(
                    create_accid(AccidType.SHARP, coord=coord, confidence=SymbolConfidence(symbol_pred, None)))
            elif label == SymbolLabel.ACCID_NATURAL:
                symbols.append(
                    create_accid(AccidType.NATURAL, coord=coord, confidence=SymbolConfidence(symbol_pred, None)))
            else:
                raise Exception("Unknown label {} during decoding".format(label))
            pass

        if False:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(5, 1, sharex='all', sharey='all')
            ax[0].imshow(p)
            ax[1].imshow(render_prediction_labels(centroids_canvas, m.region))
            labels = render_prediction_labels(p, 255 - m.region)
            ax[2].imshow(labels)
            ax[3].imshow(m.region, cmap='gray_r')
            ax[4].imshow(cc, cmap='gist_ncar_r')
            plt.show()
        line = Line(symbols=symbols)
        line.update_sequence_confidence(self.look_up)
        symbols = line.symbols

        return symbols


if __name__ == '__main__':
    from database import DatabaseBook

    b = DatabaseBook('Pa_14819')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[0:1]]

    pred = PCPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
    import matplotlib.pyplot as plt

    orig = np.array(ps[0].music_lines[0].line.operation.page_image)
    for p in ps[0].music_lines:
        for s in p.symbols:
            if s.symbol_type == SymbolType.NOTE:
                c = p.line.operation.page.page_to_image_scale(s.coord,
                                                              ref=PageScaleReference.NORMALIZED_X2).round().astype(int)
                t, l = c.y, c.x
                orig[t - 2:t + 2, l - 2:l + 2] = 255

    plt.imshow(orig)
    plt.show()
