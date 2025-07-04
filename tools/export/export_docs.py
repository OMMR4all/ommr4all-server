from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments

if __name__ == "__main__":

    book = DatabaseBook('Graduel_Syn2')
    documents = DatabaseBookDocuments().load(book)
    docs = documents.database_documents.documents
    docs_ids = []
    pages_todo = []
    for i in documents.database_documents.documents:
        t = i.get_text_of_document(book)
        for s in i.get_database_pages(book):
            if not s.page_progress().processed():
                docs_ids.append(i.monody_id)
                pages_todo.append(s.page)
                print(i.monody_id)
                print(s.page)
    docs = [i for i in docs if i.monody_id not in docs_ids]
    filename = 'CM Default Metadatendatei.xlsx'
    bytes = documents.database_documents.export_documents_to_xls(
        documents=docs,
        filename=filename,
        editor=str("Ommr4all"),
        book=book,
        flag_char="$")
    with open("/tmp/{}".format(filename), "wb") as file:
        file.write(bytes)

    docs2 = [i for i in docs if i.monody_id in docs_ids]
    filename = 'CM Default Metadatendatei_skip.xlsx'

    es = documents.database_documents.export_documents_to_xls(
        documents=docs2,
        filename=filename,
        editor=str("Ommr4all"),
        book=book,
        flag_char="$")

    with open("/tmp/{}".format(filename), "wb") as file:
        file.write(es)