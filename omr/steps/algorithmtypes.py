from enum import Enum, Enum
from typing import Dict, List


class AlgorithmTypes(Enum):
    PREPROCESSING = "preprocessing"

    STAFF_LINES_PC = "staff_lines_pc"
    STAFF_LINES_PC_Torch = "staff_lines_pc_torch"

    LAYOUT_SIMPLE_BOUNDING_BOXES = "layout_simple_bounding_boxes"
    LAYOUT_SIMPLE_LYRICS = "layout_simple_lyrics"
    LAYOUT_COMPLEX_STANDARD = "layout_complex_standard"
    LAYOUT_SIMPLE_DROP_CAPITAL = "layout_drop_capital"

    SYMBOLS_PC = "symbols_pc"
    SYMBOLS_PC_TORCH = "symbols_pc_torch"
    SYMBOLS_YOLO = "symbols_yolo"

    SYMBOLS_SEQUENCE_TO_SEQUENCE = 'symbols_sequence_to_sequence'
    SYMBOLS_SEQUENCE_TO_SEQUENCE_NAUTILUS = 'symbols_sequence_to_sequence_nautilus'

    OCR_CALAMARI = "text_calamari"
    OCR_NAUTILUS = "text_nautilus"
    OCR_GUPPY = "text_guppy"

    TEXT_DOCUMENT = "text_documents"
    TEXT_DOCUMENT_CORRECTOR = "text_documents_corrector"
    TEXT_DICTIONARY_CORRECTOR = "text_dictionary_corrector"
    TEXT_LOCALISATION = "text_localisation"

    SYLLABLES_FROM_TEXT = 'syllables_from_text'
    SYLLABLES_FROM_TEXT_TORCH = 'syllables_from_text_torch'

    SYLLABLES_IN_ORDER = 'syllables_in_order'

    DOCUMENT_ALIGNMENT = 'document_alignment'
    # Tools
    LAYOUT_CONNECTED_COMPONENTS_SELECTION = "layout_connected_components_selection"
    SYMBOLS_SEQUENCE_CONFIDENCE_CALCULATOR = "symbol_sequence_confidence_calculator"

    POSTPROCESSING = "postprocessing"

    def group(self) -> 'AlgorithmGroups':
        return [k for k, v in AlgorithmGroups.group_types_mapping().items() if self in v][0]


class AlgorithmGroups(Enum):
    PREPROCESSING = 'preprocessing'
    STAFF_LINES = 'stafflines'
    LAYOUT = 'layout'
    SYMBOLS = 'symbols'
    TEXT = 'text'
    SYLLABLES = 'syllables'
    TOOLS = 'tools'
    POSTPROCESSING = 'postprocessing'

    @staticmethod
    def group_types_mapping() -> Dict['AlgorithmGroups', List[AlgorithmTypes]]:
        return {
            AlgorithmGroups.PREPROCESSING: [AlgorithmTypes.PREPROCESSING, ],
            AlgorithmGroups.STAFF_LINES: [AlgorithmTypes.STAFF_LINES_PC, ],
            AlgorithmGroups.LAYOUT: [AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES,
                                     AlgorithmTypes.LAYOUT_COMPLEX_STANDARD, AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL],
            AlgorithmGroups.SYMBOLS: [AlgorithmTypes.SYMBOLS_PC, AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE,
                                      AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE_NAUTILUS],
            AlgorithmGroups.TEXT: [AlgorithmTypes.OCR_CALAMARI, AlgorithmTypes.OCR_NAUTILUS],
            AlgorithmGroups.SYLLABLES: [AlgorithmTypes.SYLLABLES_FROM_TEXT, AlgorithmTypes.SYLLABLES_IN_ORDER],
            AlgorithmGroups.TOOLS: [AlgorithmTypes.LAYOUT_CONNECTED_COMPONENTS_SELECTION,
                                    AlgorithmTypes.DOCUMENT_ALIGNMENT, AlgorithmTypes.TEXT_LOCALISATION],
        }

    def types(self) -> List[AlgorithmTypes]:
        return AlgorithmGroups.group_types_mapping()[self]
