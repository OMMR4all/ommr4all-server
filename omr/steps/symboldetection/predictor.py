from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple, Optional
from database import DatabasePage
from database.file_formats.pcgts import *
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResultGenerator, AlgorithmPredictionResult, PredictionCallback
from shared.pcgtscanvas import PcGtsCanvas


class SingleLinePredictionResult(NamedTuple):
    symbols: List[MusicSymbol]
    line: RegionLineMaskData

    def to_dict(self):
        return {'symbols': [s.to_json() for s in self.symbols],
                'id': self.line.operation.music_line.id}


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    pcgts: PcGts
    dataset_page: DatabasePage
    music_lines: List[SingleLinePredictionResult]
    debug_music_lines: List[SingleLinePredictionResult]

    def to_dict(self):
        return {'musicLines': [l.to_dict() for l in self.music_lines],
                'debugSymbols': [l.to_dict() for l in self.debug_music_lines]}

    def store_to_page(self):
        for line in self.music_lines:
            line.line.operation.music_line.symbols = line.symbols
        for line in self.debug_music_lines:
            line.line.operation.music_line.additional_symbols = line.symbols

        self.pcgts.page.annotations.connections.clear()
        self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class SymbolsPredictor(AlgorithmPredictor, ABC):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return all([len(l.symbols) == 0 for l in page.pcgts().page.all_music_lines()])

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        pcgts_files = [p.pcgts() for p in pages]
        page_results = [PredictionResult(pcgts, pcgts.page.location, [], []) for pcgts in pcgts_files]
        for line_results in self._predict(pcgts_files, callback):
            if isinstance(line_results, SingleLinePredictionResult):
                page_result = [p for p in page_results if p.pcgts == line_results.line.operation.pcgts][0]
                page_result.music_lines.append(line_results)
            else:
                if len(line_results) == 2:
                    line_results_symbols = line_results[0]
                    page_result = [p for p in page_results if p.pcgts == line_results_symbols.line.operation.pcgts][0]
                    page_result.music_lines.append(line_results_symbols)
                    line_results_symbols = line_results[1]
                    page_result = [p for p in page_results if p.pcgts == line_results_symbols.line.operation.pcgts][0]
                    page_result.debug_music_lines.append(line_results_symbols)

        for page_result in page_results:
            if False:
                from .evaluator import SymbolDetectionEvaluator, Codec
                evaluator = SymbolDetectionEvaluator()
                canvas = PcGtsCanvas(page_result.music_lines[0].line.operation.page, PageScaleReference.NORMALIZED_X2)
                for i, ml in enumerate(page_result.music_lines):
                    canvas.draw(ml.line.operation.music_line.staff_lines)
                    for s in ml.symbols:
                        canvas.draw_music_symbol_position_in_line(ml.line.operation.music_line.staff_lines, s, color=(255, 255, 0), thickness=2)

                    for s in ml.line.operation.music_line.symbols:
                        canvas.draw_music_symbol_position_in_line(ml.line.operation.music_line.staff_lines, s, color=(0,0, 255))

                    c = Codec()
                    gt = c.symbols_to_label_sequence(ml.line.operation.music_line.symbols, False)
                    pred = c.symbols_to_label_sequence(ml.symbols, False)
                    r = c.compute_sequence_diffs(gt, pred)
                    print(i + 1, r)
                    print(i + 1, evaluator.evaluate([ml.line.operation.music_line.symbols], [ml.symbols])[-3])

                canvas.show()
            yield page_result

    @abstractmethod
    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[SingleLinePredictionResult, None, None]:
        pass
