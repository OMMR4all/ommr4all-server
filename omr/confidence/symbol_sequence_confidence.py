import os
from enum import IntEnum
from typing import List

from database.file_formats.pcgts import MusicSymbol, SymbolType, NoteName


class ConfidenceData:
    def __init__(self, token, target):
        self.token: List[MusicSymbol] = token
        self.target: MusicSymbol = target

    def __hash__(self):
        total_string = "".join(f"{symbol.note_name}{symbol.octave}" for symbol in
                               self.token) + f"{self.target.note_name}{self.target.octave}"
        return hash(total_string)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @staticmethod
    def from_string(token, target):
        tokens = [MusicSymbol(symbol_type=SymbolType.NOTE, note_name=NoteName.from_string(singleToken[0]),
                              octave=int(singleToken[1])) for singleToken in zip(token[0::2], token[1::2])]
        targets = MusicSymbol(symbol_type=SymbolType.NOTE, note_name=NoteName.from_string(target[0]),
                              octave=int(target[1]))

        return ConfidenceData(tokens, targets)


def parse_data(path):
    from openpyxl import load_workbook
    workbook = load_workbook(path)
    workbook.active = 1
    ws = workbook.active
    target = []
    look_up_table = {}
    current_token = None
    for ind, row in enumerate(ws.values):
        for col_ind, value in enumerate(row):
            if ind == 0:
                if value != "Token":
                    target.append(value.strip())
            else:
                if col_ind == 0:
                    current_token = value
                else:
                    key = ConfidenceData.from_string(current_token.replace(" ", ""), target[col_ind - 1])
                    look_up_table[key] = value
    return look_up_table


def get_pickle_file(path):
    import pickle
    with open(path, 'rb') as file:
        unserialized_data = pickle.load(file)
    return unserialized_data


class SequenceSetting(IntEnum):
    UNDEFINED = -1
    NOTE_1GRAM = 0
    NOTE_2GRAM = 1
    NOTE_3GRAM = 2
    NOTE_4GRAM = 3

    def get_look_up(self):
        b_dir = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'ExcelTables', 'Notes')
        return parse_data([os.path.join(b_dir, 'Note_1Gram_Table.xlsx'),
                           os.path.join(b_dir, 'Note_2Gram_Table.xlsx'),
                           os.path.join(b_dir, 'Note_3Gram_Table.xlsx'),
                           os.path.join(b_dir, 'Note_4Gram_Table.xlsx')][self.value])

    def get_look_up_pickle(self):
        b_dir = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'ExcelTables', 'Notes')
        return get_pickle_file([os.path.join(b_dir, 'Note_1Gram_Table.pickle'),
                                os.path.join(b_dir, 'Note_2Gram_Table.pickle'),
                                os.path.join(b_dir, 'Note_3Gram_Table.pickle'),
                                os.path.join(b_dir, 'Note_4Gram_Table.pickle')][self.value])


class SymbolSequenceConfidence:
    def __init__(self, setting: SequenceSetting = SequenceSetting.NOTE_4GRAM):
        self.setting = setting
        self.look_up = None
        try:
            self.look_up = self.setting.get_look_up_pickle()
        except:
            self.look_up = self.setting.get_look_up()

    def get_symbol_sequence_confidence(self, prev_Symbols: List[MusicSymbol], target_symbol: MusicSymbol):
        return self.look_up[ConfidenceData(prev_Symbols, target=target_symbol)]


if __name__ == "__main__":
    from ommr4all.settings import BASE_DIR

    b_dir = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'ExcelTables', 'Notes')

    # path = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'ExcelTables', 'Notes', 'Note_1Gram_Table.xlsx')
    # parse_data(path=path)
    a = SymbolSequenceConfidence(SequenceSetting.NOTE_1GRAM)
    import pickle
    with open(os.path.join(b_dir, 'Note_1Gram_Table.pickle'), 'wb') as file:
        pickle.dump(a, file, protocol=pickle.HIGHEST_PROTOCOL)
    pass
