from enum import Enum
from typing import Dict, List


class AlgorithmTypes(Enum):
    PREPROCESSING = 1000

    STAFF_LINES_PC = 2000

    LAYOUT_SIMPLE_BOUNDING_BOXES = 3000
    LAYOUT_COMPLEX_STANDARD = 3001

    SYMBOLS_PC = 4000

    # Tools
    LAYOUT_CONNECTED_COMPONENTS_SELECTION = 8001

    def group(self) -> 'AlgorithmGroups':
        return [k for k, v in AlgorithmGroups.group_types_mapping().items() if self in v][0]


class AlgorithmGroups(Enum):
    PREPROCESSING = 'preprocessing'
    STAFF_LINES = 'stafflines'
    LAYOUT = 'layout'
    SYMBOLS = 'symbols'
    TOOLS = 'tools'

    @staticmethod
    def group_types_mapping() -> Dict['AlgorithmGroups', List[AlgorithmTypes]]:
        return {
            AlgorithmGroups.PREPROCESSING: [AlgorithmTypes.PREPROCESSING, ],
            AlgorithmGroups.STAFF_LINES: [AlgorithmTypes.STAFF_LINES_PC, ],
            AlgorithmGroups.LAYOUT: [AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES, AlgorithmTypes.LAYOUT_COMPLEX_STANDARD],
            AlgorithmGroups.SYMBOLS: [AlgorithmTypes.SYMBOLS_PC],
            AlgorithmGroups.TOOLS: [AlgorithmTypes.LAYOUT_CONNECTED_COMPONENTS_SELECTION],
        }

    def types(self) -> List[AlgorithmTypes]:
        return AlgorithmGroups.group_types_mapping()[self]
