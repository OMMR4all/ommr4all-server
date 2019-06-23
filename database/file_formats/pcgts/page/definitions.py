from enum import Enum, IntEnum


class BlockType(Enum):
    PARAGRAPH = 'paragraph'
    HEADING = 'heading'
    LYRICS = 'lyrics'
    DROP_CAPITAL = 'dropCapital'
    FOLIO_NUMBER = 'folioNumber'
    MUSIC = 'music'


class MusicSymbolPositionInStaff(IntEnum):
    UNDEFINED = -1000

    # usual notation
    SPACE_0 = 0
    LINE_0 = 1
    SPACE_1 = 2
    LINE_1 = 3
    SPACE_2 = 4
    LINE_2 = 5
    SPACE_3 = 6
    LINE_3 = 7
    SPACE_4 = 8
    LINE_4 = 9
    SPACE_5 = 10
    LINE_5 = 11
    SPACE_6 = 12
    LINE_6 = 13
    SPACE_7 = 14

    # 11th Century Notation only store up/down/equal
    UP = 101
    DOWN = 99
    EQUAL = 100

    def is_undefined(self):
        return self.value == MusicSymbolPositionInStaff.UNDEFINED

    def is_absolute(self):
        return MusicSymbolPositionInStaff.SPACE_0 <= self.value < MusicSymbolPositionInStaff.SPACE_7

    def is_relative(self):
        return MusicSymbolPositionInStaff.UP <= self.value <= MusicSymbolPositionInStaff.DOWN
