from database import DatabaseBook
from database.file_formats.pcgts import SymbolType
from omr.experimenter.documents.doc_instace_evaluator import prepare_document, filter_docs

if __name__ == "__main__":
    c = DatabaseBook('mul_2_rsync_gt')
    # excel_lines1 = evaluate_stafflines(b, c)
    docs, total, skipped = prepare_document(c, c)
    docs = filter_docs(docs, 253)

    lines = []
    clefs = []
    symbols = []
    notes = []
    accids = []
    chars = 0
    words = []
    syllabels = 0
    chants = len(docs)
    pages = []
    for i in docs:

        symbols += [t for id in i.doc.get_symbols(c)[0] for t in id]
        lines += [(t.line_id, t.page)  for t in i.doc.get_symbols(c)[1]]
        #i.doc.get_text_of_document()
        line_text2 =  i.doc.get_page_line_of_document(c)

        line_text = [i[0].sentence.syllables for i in line_text2]
        for id in line_text2:
            syllabels += len(id[0].sentence.syllables)
            chars += len(id[0].text())

    symbols_c = len(symbols)
    symbols_c_clef = len([s for s in symbols if s.symbol_type == SymbolType.CLEF])
    symbols_c_note = len([s for s in symbols if s.symbol_type == SymbolType.NOTE])
    symbols_c_accid = len([s for s in symbols if s.symbol_type == SymbolType.ACCID])
    lines = len(list(set([i[0]+" " + i[1] for i in lines])))

    print(f"Chants: {chants}")
    print(f"Lines: {lines}")
    print(f"Symbols: {symbols_c}")
    print(f"Symbols Clef: {symbols_c_clef}")
    print(f"Symbols Notes: {symbols_c_note}")
    print(f"Symbols Accids: {symbols_c_accid}")
    print(f"Chars: {chars}")
    print(f"Syllabels: {syllabels}")