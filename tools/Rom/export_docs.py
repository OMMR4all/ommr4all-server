from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments

if __name__ == "__main__":

    book = DatabaseBook('Rom_1_export')
    documents = DatabaseBookDocuments().load(book)
    docs = documents.database_documents.documents
    docs_ids = []
    pages_todo = []

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
