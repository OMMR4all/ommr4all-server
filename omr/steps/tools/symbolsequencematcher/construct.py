from enum import Enum
from typing import NamedTuple
from database.file_formats.pcgts.page.definitions import MusicSymbolPositionInStaff
from database.file_formats.pcgts.page.musicsymbol import NoteName, NoteType, SymbolType, ClefType
from typing import Union, List
from itertools import tee, islice, chain


def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)


class OperationType(Enum):
    INSERT = 1
    DELETE = 2
    REPLACE = 3


class SymbolInsertion(NamedTuple):
    PISequence: int = None
    PIS: MusicSymbolPositionInStaff = None
    SymbolName: NoteName = None
    SymbolType: SymbolType = None
    NoteType: Union[NoteType, ClefType] = None
    Type: OperationType = None


class Constructor:

    def __init__(self, prediction):
        self.prediction = prediction

    def rebuild_prediction(self, clefs_to_insert: List[SymbolInsertion]):
        from database.file_formats.pcgts.page.musicsymbol import ClefType, MusicSymbol, SymbolType
        import copy
        prediction = copy.deepcopy(self.prediction)
        last_clef = None
        counter = 0
        clef_iterator = iter(clefs_to_insert)
        current_symbol_to_insert = next(clef_iterator)
        for page in prediction:
            music_blocks = [block for block in page.pcgts.page.blocks if block.block_type == block.block_type.MUSIC]

            for block, music_line in zip(music_blocks, page.music_lines):
                inserted = 0
                deleted = 0
                for index, symbols in enumerate(previous_and_next(music_line.symbols)):
                    prev_symbol, current_symbol, next_symbol = symbols
                    ## Todo
                    if prev_symbol == None:
                        prev_symbol = current_symbol
                    if current_symbol_to_insert is not None \
                            and current_symbol_to_insert.PISequence - deleted + inserted == counter:
                        x = current_symbol.coord.x - (current_symbol.coord.x - prev_symbol.coord.x) / 2
                        point = block.lines[0].staff_lines.compute_coord_by_position_in_staff(x,
                                                                                              current_symbol_to_insert.PIS)
                        if current_symbol.symbol_type == current_symbol.symbol_type.CLEF \
                                and current_symbol.clef_type == current_symbol_to_insert.NoteType \
                                and prev_symbol.position_in_staff == current_symbol_to_insert.PIS:
                            # Same Clef, nothing to insert
                            # deleted += 1
                            counter += 1
                            try:
                                current_symbol_to_insert = next(clef_iterator)
                            except StopIteration:
                                current_symbol_to_insert = None
                                break
                            continue
                            pass

                        music_symbol = MusicSymbol(clef_type=current_symbol_to_insert.NoteType,
                                                              symbol_type=current_symbol_to_insert.SymbolType,
                                                              position_in_staff=current_symbol_to_insert.PIS,
                                                              coord=point)
                        music_line.symbols.insert(index - deleted + inserted,
                                                  music_symbol)
                        last_clef = music_symbol
                        if current_symbol.symbol_type == current_symbol.symbol_type.CLEF and current_symbol.clef_type != current_symbol_to_insert.NoteType:
                            del music_line.symbols[index + 1]
                            pass
                        else:
                            inserted += 1
                        try:
                            current_symbol_to_insert = next(clef_iterator)
                        except StopIteration:
                            current_symbol_to_insert = None
                            break

                    elif current_symbol.symbol_type == current_symbol.symbol_type.CLEF:
                        if index != 0:
                            if last_clef is not None:
                                if current_symbol.clef_type != last_clef.clef_type or current_symbol.position_in_staff \
                                        != last_clef.position_in_staff:
                                    if current_symbol_to_insert is None:
                                        pass
                                    elif current_symbol.clef_type != current_symbol_to_insert.NoteType or\
                                            current_symbol.position_in_staff != current_symbol_to_insert.PIS:
                                        del music_line.symbols[index]
                                        deleted += 1
                                        continue
                        last_clef = current_symbol

                    counter += 1
        return prediction
