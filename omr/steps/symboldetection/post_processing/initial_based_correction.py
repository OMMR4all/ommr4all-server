import logging
from typing import List
logger = logging.getLogger(__name__)

from database.file_formats.pcgts import Page, BlockType, Rect, MusicSymbol
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def correct_symbols_inside_wrong_blocks(page: Page, symbols: List[MusicSymbol]):
    drop_capitals = page.blocks_of_type([BlockType.DROP_CAPITAL, BlockType.PARAGRAPH])
    for dc in drop_capitals:
        #rec: Rect = dc.aabb
        for index, symbol in reversed(list(enumerate(symbols))):
            inside_drop_capital = dc.point_in_aabb(symbol.coord)
            if inside_drop_capital:
                for drop_capital in dc.lines:
                    a = drop_capital.coords.to_points_list()
                    drop_capital = Polygon(a)
                    inside = drop_capital.contains(Point(symbol.coord.x, symbol.coord.y))
                    if inside:
                        del symbols[index]
                        logger.info(f'Dropping Symbol {symbol.to_json()} inside wrong block: {dc.block_type}')
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
                    drop_capital = Polygon(a)
                    inside = drop_capital.contains(Point(symbol.coord.x, symbol.coord.y))
                    if inside:
                        if symbol.symbol_confidence:
                            #if symbol.symbol_confidence.symbol_prediction_confidence:
                                # print(symbol.symbol_confidence.symbol_prediction_confidence.to_json())
                            if symbol.symbol_confidence.symbol_sequence_confidence:

                                #print(symbol.symbol_confidence.symbol_sequence_confidence.to_json())
                                if symbol.symbol_confidence.symbol_sequence_confidence.confidence < 0.1:
                                    del symbols[index]
                                    #logger.info(f'Dropping Symbol {symbol.to_json()} inside wrong block: {dc.block_type}')
                                    break
                            else:
                                nmb_staff_lines = page.location.book.get_meta().numberOfStaffLines
                                if not 2 < symbol.position_in_staff < nmb_staff_lines + 1:
                                    del symbols[index]
                                    break

                        else:
                            print(symbol.position_in_staff)

    return symbols