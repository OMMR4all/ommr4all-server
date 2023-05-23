import openpyxl

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from io import BytesIO
flag_char = "$"
from openpyxl import Workbook

if __name__ == "__main__":


    book = DatabaseBook('mul_2_rsync_gt')
    documents = DatabaseBookDocuments().load(book)
    docs = documents.database_documents.documents
    docs_ids = []
    for i in documents.database_documents.documents:
        t = i.get_text_of_document(book)
        if flag_char in t:
            docs_ids.append(i.monody_id)
            print(i.monody_id)
    docs = [i for i in docs if i.monody_id not in docs_ids]
    filename = 'CM Default Metadatendatei.xlsx'
    bytes = documents.database_documents.export_documents_to_xls(
        documents=docs,
        filename=filename,
        editor=str("Ommr4all"))
    with open("/tmp/{}".format(filename), "wb") as file :
        file.write(bytes)


    pass
