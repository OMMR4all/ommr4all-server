from enum import Enum, Enum
from typing import Dict, List


class AlgorithmTypes(Enum):
    PREPROCESSING = "preprocessing"

    STAFF_LINES_PC = "staff_lines_pc"

    LAYOUT_SIMPLE_BOUNDING_BOXES = "layout_simple_bounding_boxes"
    LAYOUT_COMPLEX_STANDARD = "layout_complex_standard"

    SYMBOLS_PC = "symbols_pc"

    # Tools
    LAYOUT_CONNECTED_COMPONENTS_SELECTION = "layout_connected_components_selection"

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