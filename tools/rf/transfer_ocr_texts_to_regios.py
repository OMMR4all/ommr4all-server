import json
import os
from typing import List

import edlib

from database import DatabaseBook, DatabasePage
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats import PcGts
from database.file_formats.pcgts import Page, PageScaleReference
from database.file_formats.pcgts.page import Sentence, Annotations
from database.model import Model, MetaId
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.step import Step
from omr.steps.syllables.predictor import PredictionResult, PageMatchResult, _match_syllables_to_symbols_greedy
from omr.steps.syllables.syllablesfromtexttorch.predictor import match_text2
from omr.steps.text.hyphenation.hyphenator import CombinedHyphenator, HyphenDicts
from omr.steps.text.predictor import SingleLinePredictionResult
from shared.pcgtscanvas import PcGtsCanvas
from tools.simple_gregorianik_text_export import Lyric_info

dataset_json_file_path = "/home/alexanderh/Documents/datasets/rendered_files/"

lyric_order = ["Bv34", "A", "Y", "Kl", "Zt", "Mod", "Mp"]

lyric_system_number = "7"


def remove_upper_regions(pcgts: PcGts):
    pcgts.page.blocks = sorted(pcgts.page.blocks, key=lambda x: x.aabb.top())[-14:]
    page_db = pcgts.dataset_page()
    pcgts.page.update_reading_order()
    pcgts.to_file(page_db.file('pcgts').local_path())


def set_document_start(pcgts: PcGts):
    for block in pcgts.page.blocks:
        if block.block_type == block.block_type.LYRICS:
            for line in block.lines:
                line.document_start = True
    page_db = pcgts.dataset_page()
    pcgts.page.update_reading_order()
    pcgts.to_file(page_db.file('pcgts').local_path())


def assign_text_to_lines(db_page: DatabasePage, rendered: {}):
    hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1,
                                right=1)
    file_path = dataset_json_file_path + rendered[db_page.page]
    with open(file_path, 'r', encoding='utf-8') as f:
        t = json.load(f)
        lyric_info = Lyric_info.from_dict(t)
        latine = lyric_info.latine.replace(".", "")

    variants = []

    for i in lyric_order:
        for t in lyric_info.variants:
            if t.source == i:
                variants.append(t.latine)
                print(t.latine)
                break

    for ind, block in enumerate(db_page.pcgts().page.text_blocks()):
        for line in block.lines:
            line.sentence = Sentence.from_string(hyphen.apply_to_sentence(variants[ind].replace(".", "")))
    db_page.pcgts().page.annotations.connections.clear()
    db_page.pcgts().to_file(db_page.file('pcgts').local_path())


def assign_syllabels_to_symbols(db_page: DatabasePage, rendered: {}, book):
    meta = Step.meta(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH)
    # model = meta.best_model_for_book(book)
    from ommr4all.settings import BASE_DIR

    model = Model(
        MetaId.from_custom_path(BASE_DIR + '/storage/Graduel_Syn/models/text_guppy/2023-10-05T16:39:40/', meta.type()))
    # model = Model.from_id_str()
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    match_r = [match_text2(text_line_r) for text_line_r in ocr_r.text_lines if
               len(text_line_r.line.operation.text_line.sentence.syllables) > 0]
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict([db_page]))
    for i, p in enumerate(ps):
        print(len(p.page_match_result.match_results))
        for t in p.page_match_result.match_results:
            print(t.syllables[0].xpos)
        pmr = p.page_match_result
        canvas = PcGtsCanvas(pmr.pcgts.page, PageScaleReference.NORMALIZED_X2)
        canvas.draw(pmr.match_results, color=(25, 150, 25), background=True)
        # canvas.draw(pmr.match_results)
        # canvas.draw(p.annotations)
        canvas.show()
    pass


def assign_syllabels_to_symbols2(db_page: DatabasePage, rendered: {}, book):
    meta = Step.meta(AlgorithmTypes.OCR_GUPPY)
    # model = meta.best_model_for_book(book)
    from ommr4all.settings import BASE_DIR
    from omr.steps.text.predictor import PredictionResult as TextPredictionResult

    model = Model(
        MetaId.from_custom_path(BASE_DIR + '/storage/Graduel_Syn/models/text_guppy/2023-10-05T16:39:40/', meta.type()))
    # model = Model.from_id_str()
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    ocr_predictor = meta.create_predictor(settings)
    conf = 99999
    for i in ocr_predictor.predict([db_page]):
        ocr_r: TextPredictionResult = i
        best_pred = None

        for t in ocr_r.text_lines:
            pred = [(t, pos) for t, pos in t.text if t not in ' -']
            syls = t.line.operation.text_line.sentence.syllables
            print(pred)
            print([i.text for i in syls])
            print("".join([i.text for i in syls]))
            print("".join([i[0] for i in pred]))
            ed = edlib.align("".join([i[0] for i in pred]), "".join([i.text for i in syls]), mode="NW", task="path")
            if ed["editDistance"] < conf and len("".join([i.text for i in syls])) > 0:
                conf = ed["editDistance"]
                best_pred = t.text
            print(ed["editDistance"])
        ocr_adapted: TextPredictionResult
        text_lines: List[SingleLinePredictionResult] = []
        for t in ocr_r.text_lines:
            text_lines.append(
                SingleLinePredictionResult(text=best_pred, line=t.line, hyphenated=t.hyphenated, chars=t.chars))
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


def fill_meta_infos_to_docs(book: DatabaseBook, page: DatabasePage, rendered: {}):
    documents_loaded = DatabaseBookDocuments().load(book=book)
    docs_db = documents_loaded.database_documents
    docs = documents_loaded.database_documents.documents if documents_loaded else None
    file_path = dataset_json_file_path + rendered[page.page]
    pcgts = page.pcgts()
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        t = json.load(f)
        lyric_info = Lyric_info.from_dict(t)
    for idx, doc in enumerate(sorted(docs_db.get_documents_of_page(pcgts.page), key=lambda x: x.start.row)):
        print(doc.doc_id)
        print(doc.start.line_id)
        print(doc.start.row)
        lyric_info = Lyric_info(lyric_info.index, id=lyric_info.id, variants=lyric_info.variants,
                                latine=lyric_info.latine,
                                cantus_id=lyric_info.cantus_id, meta_info=lyric_info.meta_info,
                                initium=lyric_info.initium, genre=lyric_info.genre, url=lyric_info.url,
                                dataset_source=lyric_info.dataset_source, source=lyric_order[idx],
                                meta_infos_extended=lyric_info.meta_infos_extended)
        documents_loaded.database_documents.update_document_meta_infos(lyric_info, doc.doc_id)
    documents_loaded.to_file(book)


if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings
    rendered_json_files = {}
    for file in os.listdir(dataset_json_file_path):
        if file.endswith(".json"):
            rendered_json_files[file.replace(".json", "")] = file
            print(file)
    # get all books from database
    pages = DatabaseBook("Graduel_Syn").pages()
    pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files

    for page in pages:
        page_id = page.page
        print(page_id)
        if page_id not in rendered_json_files.keys():
            print("Page " + page_id + " not in rendered files")
        remove_upper_regions(page.pcgts())
        set_document_start(page.pcgts())
        assign_text_to_lines(page, rendered_json_files)
        assign_syllabels_to_symbols2(page, rendered_json_files, DatabaseBook("Graduel_Syn"))
        fill_meta_infos_to_docs(DatabaseBook("Graduel_Syn"), page, rendered_json_files)
        page.pcgts()
        pcgts = page.pcgts().page.all_text_lines()
