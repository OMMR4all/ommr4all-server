from enum import Enum, Enum
from typing import Dict, List, Optional

from database.file_formats.performance.pageprogress import Locks
from loguru import logger

class WorkerResource(Enum):
    """Worker resource class a task can be scheduled on (see
    restapi/operationworker): CPU workers mask CUDA, GPU workers are bound to
    a physical GPU via CUDA_VISIBLE_DEVICES."""
    CPU = 'cpu'
    GPU = 'gpu'


class AlgorithmTypes(Enum):
    PREPROCESSING = "preprocessing"

    STAFF_LINES_PC = "staff_lines_pc"
    STAFF_LINES_PC_Torch = "staff_lines_pc_torch"

    LAYOUT_SIMPLE_BOUNDING_BOXES = "layout_simple_bounding_boxes"
    LAYOUT_SIMPLE_LYRICS = "layout_simple_lyrics"
    LAYOUT_COMPLEX_STANDARD = "layout_complex_standard"
    LAYOUT_SIMPLE_DROP_CAPITAL = "layout_drop_capital"
    LAYOUT_SIMPLE_DROP_CAPITAL_YOLO = "layout_drop_capital_yolo"

    SYMBOLS_PC = "symbols_pc"
    SYMBOLS_PC_TORCH = "symbols_pc_torch"
    SYMBOLS_YOLO = "symbols_yolo"

    SYMBOLS_SEQUENCE_TO_SEQUENCE = 'symbols_sequence_to_sequence'
    SYMBOLS_SEQUENCE_TO_SEQUENCE_NAUTILUS = 'symbols_sequence_to_sequence_nautilus'
    SYMBOLS_SEQUENCE_TO_SEQUENCE_GUPPY = 'symbols_sequence_to_sequence_guppy'

    OCR_CALAMARI = "text_calamari"
    OCR_NAUTILUS = "text_nautilus"
    OCR_GUPPY = "text_guppy"
    OCR_LLM = "text_llm"

    TEXT_DOCUMENT = "text_documents"
    TEXT_DOCUMENT_CORRECTOR = "text_documents_corrector"
    TEXT_DICTIONARY_CORRECTOR = "text_dictionary_corrector"
    TEXT_LOCALISATION = "text_localisation"

    SYLLABLES_FROM_TEXT = 'syllables_from_text'
    SYLLABLES_FROM_TEXT_TORCH = 'syllables_from_text_torch'

    SYLLABLES_IN_ORDER = 'syllables_in_order'

    END2END_SWIN = "end2end_swin"

    DOCUMENT_ALIGNMENT = 'document_alignment'
    # Tools
    LAYOUT_CONNECTED_COMPONENTS_SELECTION = "layout_connected_components_selection"
    SYMBOLS_SEQUENCE_CONFIDENCE_CALCULATOR = "symbol_sequence_confidence_calculator"
    SYMBOLS_PATTERN_MATCHER =  "symbols_pattern_matcher"

    POSTPROCESSING = "postprocessing"

    def group(self) -> 'AlgorithmGroups':
        groups = [k for k, v in AlgorithmGroups.group_types_mapping().items() if self in v]
        if not groups:
            # Every type should be mapped; falling back to TOOLS (which carries no lock and
            # no default model) keeps callers such as AlgorithmPredictor.unlocked working
            # instead of raising for a type someone forgot to register.
            logger.warning("AlgorithmType {} is not assigned to a group".format(self))
            return AlgorithmGroups.TOOLS
        return groups[0]

    def uses_model(self) -> bool:
        """Whether this step loads a model from a model directory.

        Only these types have a default model per book style (see
        restapi/views/administrativedefaultmodels.py). Rule based steps (syllables in order),
        pure tools, pre-/postprocessing and the remotely served LLM transcription do not.
        Note that several types may share one model directory, see model_type().
        """
        return self in {
            AlgorithmTypes.STAFF_LINES_PC,
            AlgorithmTypes.STAFF_LINES_PC_Torch,

            AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES,
            AlgorithmTypes.LAYOUT_COMPLEX_STANDARD,
            AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL,
            AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO,
            AlgorithmTypes.LAYOUT_SIMPLE_LYRICS,

            AlgorithmTypes.SYMBOLS_PC,
            AlgorithmTypes.SYMBOLS_PC_TORCH,
            AlgorithmTypes.SYMBOLS_YOLO,
            AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE,
            AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE_NAUTILUS,
            AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE_GUPPY,

            AlgorithmTypes.OCR_CALAMARI,
            AlgorithmTypes.OCR_NAUTILUS,
            AlgorithmTypes.OCR_GUPPY,

            AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH,

            AlgorithmTypes.END2END_SWIN,
        }

    def model_type(self):
        return {
            AlgorithmTypes.LAYOUT_SIMPLE_LYRICS: AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO,
            AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH: AlgorithmTypes.OCR_GUPPY,
        }[self] if self in {
            AlgorithmTypes.LAYOUT_SIMPLE_LYRICS,
            AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH,
        } else self



class AlgorithmGroups(Enum):
    PREPROCESSING = 'preprocessing'
    STAFF_LINES = 'stafflines'
    LAYOUT = 'layout'
    SYMBOLS = 'symbols'
    TEXT = 'text'
    SYLLABLES = 'syllables'
    END2END = 'end2end'
    TOOLS = 'tools'
    POSTPROCESSING = 'postprocessing'

    @staticmethod
    def group_types_mapping() -> Dict['AlgorithmGroups', List[AlgorithmTypes]]:
        # Every AlgorithmTypes member must appear exactly once: group() derives the page
        # lock (see group_2_lock_mapping) from this mapping, so a missing type would make
        # the page selection blind to that step's locks. Deactivated/legacy types are
        # appended at the end of their group because AlgorithmGroups.types()[0] is used as
        # the group's default model type (restapi/views/administrativedefaultmodels.py).
        return {
            AlgorithmGroups.PREPROCESSING: [AlgorithmTypes.PREPROCESSING, ],
            AlgorithmGroups.STAFF_LINES: [AlgorithmTypes.STAFF_LINES_PC_Torch,
                                          AlgorithmTypes.STAFF_LINES_PC],
            AlgorithmGroups.LAYOUT: [AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES,
                                     AlgorithmTypes.LAYOUT_COMPLEX_STANDARD, AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL, AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO,
                                     AlgorithmTypes.LAYOUT_SIMPLE_LYRICS],
            AlgorithmGroups.SYMBOLS: [AlgorithmTypes.SYMBOLS_PC_TORCH, AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE_GUPPY,
                                      AlgorithmTypes.SYMBOLS_PC, AlgorithmTypes.SYMBOLS_YOLO,
                                      AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE,
                                      AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE_NAUTILUS],
            AlgorithmGroups.TEXT: [AlgorithmTypes.OCR_GUPPY, AlgorithmTypes.OCR_LLM, AlgorithmTypes.TEXT_DOCUMENT,
                                   AlgorithmTypes.OCR_CALAMARI, AlgorithmTypes.OCR_NAUTILUS,
                                   AlgorithmTypes.TEXT_DOCUMENT_CORRECTOR, AlgorithmTypes.TEXT_DICTIONARY_CORRECTOR],
            AlgorithmGroups.SYLLABLES: [AlgorithmTypes.SYLLABLES_IN_ORDER,
                                        AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH,
                                        AlgorithmTypes.SYLLABLES_FROM_TEXT],
            AlgorithmGroups.END2END: [AlgorithmTypes.END2END_SWIN],
            AlgorithmGroups.TOOLS: [AlgorithmTypes.LAYOUT_CONNECTED_COMPONENTS_SELECTION,
                                    AlgorithmTypes.DOCUMENT_ALIGNMENT, AlgorithmTypes.TEXT_LOCALISATION, AlgorithmTypes.SYMBOLS_PATTERN_MATCHER,
                                    AlgorithmTypes.SYMBOLS_SEQUENCE_CONFIDENCE_CALCULATOR],
            AlgorithmGroups.POSTPROCESSING: [AlgorithmTypes.POSTPROCESSING],
        }

    def types(self) -> List[AlgorithmTypes]:
        return AlgorithmGroups.group_types_mapping()[self]

    def group_2_lock_mapping(self) -> Optional[Locks]:
        # None means "this group has no page lock", i.e. it never overwrites annotations a
        # user could have locked. Unmapped groups degrade to None rather than raising.
        return {
            AlgorithmGroups.PREPROCESSING: None,
            AlgorithmGroups.STAFF_LINES: Locks.STAFF_LINES,
            AlgorithmGroups.LAYOUT: Locks.LAYOUT,
            AlgorithmGroups.SYMBOLS: Locks.SYMBOLS,
            AlgorithmGroups.TEXT: Locks.TEXT,
            AlgorithmGroups.SYLLABLES: Locks.TEXT,
            AlgorithmGroups.END2END: Locks.SYMBOLS,
            AlgorithmGroups.TOOLS: None,
            AlgorithmGroups.POSTPROCESSING: None,
        }.get(self)
