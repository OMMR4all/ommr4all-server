import os
from typing import List

import edlib

from database import DatabasePage, DatabaseBook
from database.file_formats.pcgts.page import Annotations
from database.model import Model, MetaId
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.step import Step
from omr.steps.syllables.predictor import PageMatchResult, _match_syllables_to_symbols_greedy, PredictionResult
from omr.steps.syllables.syllablesfromtexttorch.predictor import match_text2
from omr.steps.text.predictor import SingleLinePredictionResult


def assign_syllabels_to_symbols(db_page: DatabasePage, rendered: {}, book):
    meta = Step.meta(AlgorithmTypes.OCR_GUPPY)
    # model = meta.best_model_for_book(book)
    from ommr4all.settings import BASE_DIR
    from omr.steps.text.predictor import PredictionResult as TextPredictionResult

    model = Model(
        MetaId.from_custom_path(BASE_DIR + '/storage/Rom_1/models/text_guppy/2025-08-09T13:35:57/', meta.type()))
    # model = Model.from_id_str()
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    ocr_predictor = meta.create_predictor(settings)
    for i in ocr_predictor.predict([db_page]):
        ocr_r: TextPredictionResult = i
        text_lines: List[SingleLinePredictionResult] = []

        for ind1, y in enumerate(ocr_r.text_lines):
            best_pred = None
            ind_e = None
            conf = 99999

            for ind, t in enumerate(ocr_r.text_lines):
                pred = [(t, pos) for t, pos in t.text if t not in ' -']
                syls = y.line.operation.text_line.sentence.syllables
                #print(pred)
                #print([i.text for i in syls])
                #print("".join([i.text for i in syls]))
                #print("".join([i[0] for i in pred]))
                print(":::")
                print("".join([i[0] for i in pred]))
                print("".join([i.text for i in syls]))
                ed = edlib.align("".join([i[0] for i in pred]), "".join([i.text for i in syls]), mode="NW", task="path")
                if ed["editDistance"] < conf and len("".join([i.text for i in syls])) > 0:
                    conf = ed["editDistance"]
                    best_pred = t.text
                    ind_e = ind
            #print(ed["editDistance"])
            if best_pred == None:
                print(y.line.operation.page.location.page)
            text_lines.append(
                SingleLinePredictionResult(text=best_pred, line=y.line, hyphenated=y.hyphenated, chars=y.chars))
        ocr_adapted: TextPredictionResult

        ocr_adapted = TextPredictionResult(ocr_r.pcgts, ocr_r.dataset_page, text_lines)

        match_r = [match_text2(text_line_r) for text_line_r in ocr_adapted.text_lines if
                   len(text_line_r.line.operation.text_line.sentence.syllables) > 0]
        p_result = PageMatchResult(text_prediction_result=ocr_r, match_results=match_r, pcgts=ocr_r.pcgts)
        annotations = Annotations(p_result.page())
        for mr in p_result.match_results:
            # self._match_syllables_to_symbols_bipartite_matching(mr, pr.page(), annotations)
            # self._match_syllables_to_symbols(mr, pr.page(), annotations)
            _match_syllables_to_symbols_greedy(mr, p_result.page(), annotations)
        prediction_result = PredictionResult(
            page_match_result=p_result,
            annotations=annotations)
        prediction_result.store_to_page()

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings

    # get all books from database
    book = DatabaseBook("Rom_1")
    pages = book.pages()
    lines = None
    with open("groupings.csv", "r") as csv_file:
        lines_ = csv_file.readlines()
        lines = [line.strip() for line in lines_]
    # pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files
    excepted_ids = []
    for page, line in zip(pages, lines):
        assign_syllabels_to_symbols(page, None, book)
