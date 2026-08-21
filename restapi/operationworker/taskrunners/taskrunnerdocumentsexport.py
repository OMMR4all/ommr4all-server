import base64
import io
import json
import zipfile
from typing import Tuple

from database import DatabaseBook, DatabasePage
from database.file_formats.exporter.document_export import (
    document_file_name, mei_files_of_document, monodi_json_of_document,
)
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

    def run(self, task: Task, com_queue: Queue) -> dict:
        from database.database_book_documents import DatabaseBookDocuments
        # refresh rather than load: an export must never emit documents assembled by an
        # older version of the code (cheap when nothing changed)
        documents = DatabaseBookDocuments.update_book_documents_cached(self.book).database_documents.documents
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
            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, doc in enumerate(documents):
                    json_data = monodi_json_of_document(self.book, doc, editor)
                    zf.writestr(document_file_name(doc) + '.json', json.dumps(json_data, indent=2))
                    progress(i + 1)
            data = s.getvalue()
            filename = self.book.book + '.monodiplus.zip'
            mime = 'application/zip'
        elif self.export_format == TaskRunnerDocumentsExport.FORMAT_MEI4_ZIP:
            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, doc in enumerate(documents):
                    doc_dir = document_file_name(doc)
                    # no page cache here: the MEI of a page now depends on the document
                    for page_name, xml in mei_files_of_document(self.book, doc):
                        zf.writestr(doc_dir + '/' + page_name + '.xml', xml)
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
