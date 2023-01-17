import random
from tkinter import Image
from typing import List

import numpy as np
import shapely

from database.file_formats.pcgts import MusicSymbol, Page, Point, MusicSymbolPositionInStaff, StaffLines, create_clef, \
    ClefType
from database.file_formats.pcgts.page import SymbolErrorType
from omr.dataset import RegionLineMaskData

from database.file_formats.pcgts import Page, BlockType, Rect, MusicSymbol
import loguru
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
                    # value =
                    while True:
                        coord = Point(x=coord.x, y=coord.y + 0.001)
                        new_pos = m.operation.music_line.compute_position_in_staff(coord)
                        if pos != new_pos:
                            symbol_closer_to_staff_line.coord = coord
                            break
            prev_symbol = symbol
    return symbols


def fix_pos_of_close_symbols2(page, symbols: List[MusicSymbol], scale_reference, debug=False, m=None):
    if len(symbols) > 0:
        avg_line_distance = page.avg_staff_line_distance()
        avg_line_distance_step = avg_line_distance * 3 / 50

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
            symbols_new = []
            for x in symbols:
                if x.position_in_staff == pis:
                    symbols_new.append(x)
            return symbols_new

        # print("newline")
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
                # print(coord.y)
                new_pis = m.operation.music_line.compute_position_in_staff(coord)
                if pis != new_pis:
                    if its == 0:
                        # symbol.note_type = symbol.note_type.APOSTROPHA
                        prev_symbol = None
                        next_symbol = None
                        distance_1 = None
                        distance_2 = None
                        if ind > 1:
                            prev_symbol = symbols[ind - 1]
                        if ind < len(symbols) - 1:
                            next_symbol = symbols[ind + 1]
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
                            # symbol.note_type = symbol.note_type.ORISCUS
                            symbol.coord = coord
                            symbol.position_in_staff = new_pis

                        # symbol.coord= coord
                    # print(its)

                    break
                if its > 1:
                    break
                its += 1

    return symbols

def fix_pos_of_close_symbols3(page, symbols: List[MusicSymbol], scale_reference, debug=False, m=None):
    if len(symbols) > 0:
        avg_line_distance = page.avg_staff_line_distance()
        avg_line_distance_step = avg_line_distance * 4 / 50

        def distance_bet_symbols(symbol1: MusicSymbol, symbol2: MusicSymbol):
            distance = symbol2.coord.x - symbol1.coord.x
            return distance

        prev_symbol =symbols[0]
        prev_symbol_snapped_pos = m.operation.music_line.staff_lines.snap_to_pos(prev_symbol.coord)
        found = False
        for ind, symbol in enumerate(symbols[1:]):
            if found:
                found = False
                prev_symbol = symbol
                prev_symbol_snapped_pos = m.operation.music_line.staff_lines.snap_to_pos(symbol.coord)
                continue
            snapped_pos = m.operation.music_line.staff_lines.snap_to_pos(symbol.coord)
            distance_to_snap = abs(symbol.coord.y - snapped_pos)
            distance_to_snap_prev_symbol = abs(prev_symbol.coord.y - prev_symbol_snapped_pos)

            coord = None
            its = 0
            if distance_to_snap_prev_symbol > distance_to_snap:
                symbol_to_check = prev_symbol
                snap_to_check = prev_symbol_snapped_pos
                other = symbol
            else:
                symbol_to_check = symbol
                snap_to_check = snapped_pos
                other = prev_symbol
            coord = symbol_to_check.coord
            pis = symbol_to_check.position_in_staff

            while True:

                if snap_to_check > symbol_to_check.coord.y:
                    coord = Point(x=coord.x, y=coord.y - avg_line_distance_step)
                else:
                    coord = Point(x=coord.x, y=coord.y + avg_line_distance_step)
                # print(coord.y)
                new_pis = m.operation.music_line.compute_position_in_staff(coord)
                if pis != new_pis:
                    if its == 0:
                        #symbol_to_check.note_type = symbol.note_type.APOSTROPHA
                        distance = distance_bet_symbols(prev_symbol, symbol)
                        if distance < avg_line_distance / 2 and abs(other.position_in_staff.value - new_pis.value) == 1:
                            #symbol_to_check.note_type = symbol_to_check.note_type.ORISCUS
                            symbol_to_check.coord = coord
                            symbol_to_check.position_in_staff = new_pis
                            found = True

                        # symbol.coord= coord
                    # print(its)

                    break
                if its > 1:
                    break
                its += 1
            prev_symbol = symbol
            prev_symbol_snapped_pos = m.operation.music_line.staff_lines.snap_to_pos(symbol.coord)

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
                                    symbols_between[0][0].graphical_connection == symbols_between[0][
                                0].graphical_connection.NEUME_START:

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



def correct_symbols_inside_wrong_blocks(page: Page, symbols: List[MusicSymbol]):
    drop_capitals = page.blocks_of_type([BlockType.DROP_CAPITAL, BlockType.PARAGRAPH])
    for dc in drop_capitals:
        #rec: Rect = dc.aabb
        for index, symbol in reversed(list(enumerate(symbols))):
            inside_drop_capital = dc.point_in_aabb(symbol.coord)
            if inside_drop_capital:
                for drop_capital in dc.lines:
                    a = drop_capital.coords.to_points_list()
                    drop_capital = shapely.geometry.polygon.Polygon(a)
                    inside = drop_capital.contains(shapely.geometry.Point(symbol.coord.x, symbol.coord.y))
                    if inside:
                        del symbols[index]
                        break
    return symbols

def correct_symbols_inside_text_blocks(page: Page, symbols: List[MusicSymbol]):
    drop_capitals = page.blocks_of_type([BlockType.LYRICS])
    for dc in drop_capitals:
        #rec: Rect = dc.aabb
        for index, symbol in reversed(list(enumerate(symbols))):
            symbol: MusicSymbol =symbol
            inside_drop_capital = dc.point_in_aabb(symbol.coord)
            if inside_drop_capital:
                for drop_capital in dc.lines:
                    a = drop_capital.coords.to_points_list()
                    drop_capital = shapely.geometry.polygon.Polygon(a)
                    inside = drop_capital.contains(shapely.geometry.Point(symbol.coord.x, symbol.coord.y))
                    if inside:
                        if symbol.symbol_confidence:
                            #if symbol.symbol_confidence.symbol_prediction_confidence:
                                # print(symbol.symbol_confidence.symbol_prediction_confidence.to_json())
                            if symbol.symbol_confidence.symbol_sequence_confidence:

                                #print(symbol.symbol_confidence.symbol_sequence_confidence.to_json())
                                if symbol.symbol_confidence.symbol_sequence_confidence.confidence < 0.1:
                                    del symbols[index]
                                    loguru.logger.info(f'Dropping Symbol {symbol.to_json()} inside wrong block: {dc.block_type}')
                                    break
                            else:
                                nmb_staff_lines = page.location.book.get_meta().numberOfStaffLines
                                if not 2 < symbol.position_in_staff < nmb_staff_lines + 1:
                                    del symbols[index]
                                    break

                        else:
                            pass
                            #print(symbol.position_in_staff)

    return symbols