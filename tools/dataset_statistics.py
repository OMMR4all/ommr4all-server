import symbol
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json, LetterCase
from prettytable import PrettyTable
from shapely.geometry import Polygon, Point

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import SymbolType, MusicSymbol, Line, Block, BlockType
from database.tools.book_statistics import compute_book_statistics


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Counts:
    n_pages: int = 0
    n_staves: int = 0
    n_staff_lines: int = 0
    n_symbols: int = 0
    n_note_components: int = 0
    n_clefs: int = 0
    n_accids: int = 0
    n_symbols_in_text_area: int = 0
    n_clef_after_drop_capital_area_all: int = 0
    n_clef_after_drop_capital_area_big: int = 0
    n_symbol_in_drop_capital_area: int = 0
    n_symbol_above_para_text_area: int = 0
    n_drop_capitals_all: int = 0
    n_drop_capitals_big: int = 0
    n_para_text: int = 0
    n_para_text2: int = 0
    n_para_drop_captial2: int = 0



    def drop_capitals_of_text_line(self, tl: Line, max_y_difference=0.001) -> List[Block]:
        drop_capitals = self.blocks_of_type(BlockType.DROP_CAPITAL)
        line_dp_capitals = []
        for d_c in drop_capitals:
            if tl.aabb.bottom() < d_c.aabb.bottom() < tl.aabb.top():  # or abs(d_c.aabb.bottom() - tl.aabb.bottom() < max_y_difference):
                line_dp_capitals.append(d_c)
        return line_dp_capitals

    def para_text_of_text_line(self, tl: Line, max_y_difference=0.001) -> List[Block]:
        drop_capitals = self.blocks_of_type(BlockType.PARAGRAPH)
        line_dp_capitals = []
        for d_c in drop_capitals:

            if tl.aabb.bottom() < d_c.aabb.bottom() < tl.aabb.top():  # or abs(d_c.aabb.bottom() - tl.aabb.bottom() < max_y_difference):
                line_dp_capitals.append(d_c)
        return line_dp_capitals


class Callback(ABC):
    @abstractmethod
    def updated(self, counts: Counts, n_processed, n_total):
        pass

def drop_captial_of_text_line_center(tl: Line, ml: Line, drop_capitals: List[Block]):
    caps = []
    if len(drop_capitals) == 0:
        return []
    else:
        u_aabb = tl.aabb.union(ml.aabb)
        for d_c in drop_capitals:


            if u_aabb.top() < d_c.aabb.center.y < u_aabb.bottom():
                caps.append(d_c)
    return caps

def para_text_of_text_line_center(tl: Line, ml: Line, drop_capitals: List[Block]):
    caps = []
    if len(drop_capitals) == 0:
        return []
    else:
        u_aabb = tl.aabb.union(ml.aabb)
        for d_c in drop_capitals:
            if u_aabb.top() < d_c.aabb.center.y < u_aabb.bottom():
                caps.append(d_c)
    return caps
def compute_books_statistics(books: List[DatabaseBook], ignore_page: List[str] = None,
                             callback: Callback = None) -> Counts:
    ignore_page = ignore_page if ignore_page else []
    pages = sum(
        [[page.pcgts() for page in book.pages() if not any([s in page.page for s in ignore_page])] for book in books],
        [])
    return get_counts(pages, callback)


def compute_book_statistics(book: DatabaseBook, ignore_page: List[str] = None, callback: Callback = None) -> Counts:
    return compute_books_statistics([book], ignore_page, callback)


def symbols_between_x1_x2(symbols: List[MusicSymbol], x1, x2) -> List[MusicSymbol]:
    between = []
    for symbol in symbols:
        if x1 < symbol.coord.x < x2:
            between.append(symbol)
    return between


def symbols_in_line(symbols: List[MusicSymbol], line: Line):
    symbols_in_line = []
    if line:
        a = line.coords.to_points_list()
        drop_capital = Polygon(a)
        for symbol in symbols:
            inside = drop_capital.contains(Point(symbol.coord.x, symbol.coord.y))
            if inside:
                symbols_in_line.append(symbol)
    else:
        print("D")
    return symbols_in_line


def symbols_in_block(symbols: List[MusicSymbol], line: Block):
    symbols_in_line = []
    a = line.coords.to_points_list()
    drop_capital = Polygon(a)
    for symbol in symbols:
        inside = drop_capital.contains(Point(symbol.coord.x, symbol.coord.y))
        if inside:
            symbols_in_line.append(symbol)
    return symbols_in_line


def get_counts(pages: List[PcGts], callback: Callback = None) -> Counts:
    counts = Counts()

    if callback:
        callback.updated(counts, 0, len(pages))

    for i, pcgts in enumerate(pages):
        p = pcgts.page
        counts.n_pages += 1
        mls = p.all_music_lines()
        counts.n_staves += len(mls)
        paragraph = p.blocks_of_type(BlockType.PARAGRAPH)
        drop_capitals = p.blocks_of_type(BlockType.DROP_CAPITAL)
        counts.n_para_drop_captial2 += len(drop_capitals)

        counts.n_para_text2 += len(p.blocks_of_type(BlockType.PARAGRAPH))
        #print(pcgts.dataset_page().page)
        for ind, ml in enumerate(mls):
            b_tl = p.closest_below_text_line_to_music_line(ml, True)
            a_tl = p.closest_above_text_line_to_music_line(ml, True)
            d_capitals = drop_captial_of_text_line_center(b_tl, ml, drop_capitals) ## p.drop_capitals_of_text_line(b_tl)
            para_text = para_text_of_text_line_center(b_tl, ml, paragraph)
            #para_text = p.para_text_of_text_line(b_tl)
            symbols_above_para_text = []
            for para in para_text:
                ss = symbols_between_x1_x2(ml.symbols, para.aabb.left(), para.aabb.right())
                #if len(ss) > 0:
                #    print(p.location.page)

                symbols_above_para_text += ss

            def symbol_after_x(x, symbols: List[MusicSymbol]):
                for symbol_1 in symbols:
                    if symbol_1.coord.x > x:
                        return symbol_1
                return None
            for cap in d_capitals:
                center_x = cap.aabb.center.x

                next_symbol = symbol_after_x(center_x, ml.symbols)
                if next_symbol is not None:
                    if next_symbol.symbol_type == SymbolType.CLEF:
                        counts.n_clef_after_drop_capital_area_all += 1
                        if cap.aabb.bottom() - cap.aabb.top() > 0.09:
                            counts.n_clef_after_drop_capital_area_big += 1
                            counts.n_drop_capitals_big += 1

            symbols_in_text_area = symbols_in_line(ml.symbols, a_tl)
            symbols_in_text_area += symbols_in_line(ml.symbols, b_tl)
            #if len(symbols_in_text_area) > 0:
            #    print(pcgts.dataset_page().page)
            symbols_in_drop_capital = []
            for d in d_capitals:
                symbols_in_drop_capital += symbols_in_block(ml.symbols, d)

            counts.n_staff_lines += len(ml.staff_lines)
            counts.n_symbols += len(ml.symbols)
            counts.n_note_components += len([s for s in ml.symbols if s.symbol_type == SymbolType.NOTE])
            counts.n_clefs += len([s for s in ml.symbols if s.symbol_type == SymbolType.CLEF])
            counts.n_accids += len([s for s in ml.symbols if s.symbol_type == SymbolType.ACCID])
            counts.n_drop_capitals_all += len(d_capitals)
            counts.n_para_text += len(para_text)
            counts.n_symbol_above_para_text_area += len(symbols_above_para_text)
            counts.n_symbols_in_text_area += len(symbols_in_text_area)
            counts.n_symbol_in_drop_capital_area += len(symbols_in_drop_capital)

        if callback:
            callback.updated(counts, i + 1, len(pages))

    return counts


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--books", nargs="+", required=True)
    parser.add_argument("--ignore-page", nargs="+", default=[])

    args = parser.parse_args()

    all_book_counts = [compute_book_statistics(DatabaseBook(book), args.ignore_page) for book in args.books]

    pt = PrettyTable([n for n, _ in all_book_counts[0].to_dict().items()])
    for book_counts in all_book_counts:
        pt.add_row([v for _, v in book_counts.to_dict().items()])

    print(pt)
