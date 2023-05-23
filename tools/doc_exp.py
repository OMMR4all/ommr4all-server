mid= "2a6ca93b-53ed-41fb-9d1f-8b86e2ad8515"

import json

import openpyxl
from openpyxl import Workbook

from database import DatabaseBook, DatabasePage, DatabaseFile
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter

book = DatabaseBook('mul_2_rsync_gt')
documents = DatabaseBookDocuments().load(book)

document = documents.database_documents.get_document_by_monodi_id(mid)
pages = [DatabasePage(book, x) for x in document.pages_names]
pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]
root = PcgtsToMonodiConverter(pcgts, document=document)
json_data = root.get_Monodi_json(document=document, editor=str("OMMR4all"))

t = json.dumps(json_data)
print(t)