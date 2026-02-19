import os

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.pcgts.page import Sentence

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings
    rendered_json_files = {}

    # get all books from database
    book = DatabaseBook("Köln_Dombibl_1001b_cp")
    book2 = DatabaseBook("Köln_Dombibl_1001b_cp_new_pred")

    documents = DatabaseBookDocuments().load(book)

    id_with_flag = []
    id_without_flag = []
    all_ids = []
    for document in documents:
        text = document.get_text_of_document(book)
        id = document.doc_id
        all_ids.append(id)
        for i in ["$", "%", "&"]:
            if i in text:
                print(f"Document {document.monody_id} contains invalid characters. Char {i}")
                id_with_flag.append(id)
    id_without_flag = set(all_ids) - set(id_with_flag)

    for i in all_ids:
        if i in id_with_flag:
            #docs1 = documents.database_documents.get_document_by_id(i, book)
            docs2 = documents.database_documents.get_document_by_id(i)
            pages = []
            first = True
            for pairs in zip(docs2.get_page_line_of_document(book), docs2.get_page_line_of_document(book2)):

                pair = pairs[0]
                pair2 = pairs[1]
                line = pair[0]
                page = pair[1]
                line2 = pair2[0]
                page2 = pair2[1]
                pages.append(page)
                if first:
                    hyph = "$ "+ line2.sentence.text()
                    first = False
                else:
                    hyph = line2.sentence.text()
                #line.line.operation.text_line.sentence = Sentence.from_string(line.hyphenated)

                line.sentence = Sentence.from_string(hyph)
            for page in set(pages):
                page.pcgts().to_file(page.file('pcgts').local_path())
                print(f"Updated page {page.page} in document {i}: local_path={page.file('pcgts').local_path()}")


