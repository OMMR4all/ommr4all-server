import base64
import io
import json
import re
import zipfile
from typing import Tuple

from database import DatabaseBook, DatabasePage, DatabaseFile
from .pageselection import PageSelection
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from ..taskcommunicator import TaskCommunicationData
import logging

logger = logging.getLogger(__name__)


class TaskRunnerDocumentsExport(TaskRunner):
    FORMAT_MONODI_META_XLSX = 'monodi_meta.xlsx'
    FORMAT_MONODI_PLUS_ZIP = 'monodiplus.zip'
    FORMAT_MEI4_ZIP = 'mei4.zip'
    FORMATS = [FORMAT_MONODI_META_XLSX, FORMAT_MONODI_PLUS_ZIP, FORMAT_MEI4_ZIP]

    def __init__(self,
                 book: DatabaseBook,
                 export_format: str,
                 ):
        super().__init__(None, PageSelection.from_book(book), [TaskWorkerGroup.SHORT_TASKS_CPU])
        self.book = book
        self.export_format = export_format

    def identifier(self) -> Tuple:
        return self.book.book, self.export_format

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return True

    @staticmethod
    def _document_file_name(doc) -> str:
        initium = doc.document_meta_infos.initium if doc.document_meta_infos and doc.document_meta_infos.initium else doc.textinitium
        initium = (initium or '').replace('-', '')
        initium = re.sub(r'[^\w]+', '_', initium).strip('_')[:60]
        return '_'.join(filter(None, [initium, doc.doc_id]))

    def _load_pcgts_of_document(self, doc):
        pages = [DatabasePage(self.book, name) for name in doc.pages_names]
        return [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]

    def run(self, task: Task, com_queue: Queue) -> dict:
        from database.database_book_documents import DatabaseBookDocuments
        documents = DatabaseBookDocuments().load(self.book).database_documents.documents
        editor = str(task.creator.username) if task.creator else ''
        n_total = len(documents)

        def progress(n_processed: int):
            com_queue.put(TaskCommunicationData(task, TaskStatus(
                TaskStatusCodes.RUNNING,
                TaskProgressCodes.WORKING,
                progress=n_processed / n_total if n_total > 0 else 1,
                n_processed=n_processed,
                n_total=n_total,
            )))

        progress(0)

        if self.export_format == TaskRunnerDocumentsExport.FORMAT_MONODI_META_XLSX:
            from database.file_formats.book.documents import Documents
            data = Documents.export_documents_to_xls(
                documents=documents,
                filename='CM Default Metadatendatei',
                editor=editor,
                book=self.book,
                callback=progress)
            filename = self.book.book + '.monodi_meta.xlsx'
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif self.export_format == TaskRunnerDocumentsExport.FORMAT_MONODI_PLUS_ZIP:
            from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, doc in enumerate(documents):
                    pcgts = self._load_pcgts_of_document(doc)
                    root = PcgtsToMonodiConverter(pcgts, document=doc)
                    json_data = root.get_Monodi_json(document=doc, editor=editor)
                    zf.writestr(self._document_file_name(doc) + '.json', json.dumps(json_data, indent=2))
                    progress(i + 1)
            data = s.getvalue()
            filename = self.book.book + '.monodiplus.zip'
            mime = 'application/zip'
        elif self.export_format == TaskRunnerDocumentsExport.FORMAT_MEI4_ZIP:
            from database.file_formats.exporter.mei.pcgts_to_mei4_exporter import PcgtsToMeiConverter
            page_cache = {}

            def mei_of_page(page_name: str) -> str:
                if page_name not in page_cache:
                    page = DatabasePage(self.book, page_name)
                    pcgts = DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts()
                    page_cache[page_name] = PcgtsToMeiConverter(pcgts).to_string()
                return page_cache[page_name]

            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, doc in enumerate(documents):
                    doc_dir = self._document_file_name(doc)
                    for page_name in doc.pages_names:
                        zf.writestr(doc_dir + '/' + page_name + '.xml', mei_of_page(page_name))
                    progress(i + 1)
            data = s.getvalue()
            filename = self.book.book + '.mei.zip'
            mime = 'application/zip'
        else:
            raise ValueError('Unknown export format: {}'.format(self.export_format))

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.FINALIZING)))
        return {
            'filename': filename,
            'mime': mime,
            'data': base64.b64encode(data).decode('ascii'),
        }
