from abc import ABC, abstractmethod
from typing import List, Tuple, Type
from multiprocessing import Queue
from restapi.operationworker.taskrunners.pageselection import PageSelection
from ..task import Task
from ..taskworkergroup import TaskWorkerGroup
from database.database_available_models import DatabaseAvailableModels, DefaultModel
from database.database_page import DatabasePage, DatabaseBook
from omr.steps.step import Step, AlgorithmTypes, AlgorithmMeta
from omr.steps.algorithmtypes import AlgorithmGroups


class TaskRunner(ABC):
    def __init__(self,
                 algorithm_type: AlgorithmTypes,
                 selection: PageSelection,
                 task_group: List[TaskWorkerGroup]):
        self.algorithm_type = algorithm_type
        self.selection: PageSelection = selection
        self.task_group = task_group

    def algorithm_meta(self) -> Type[AlgorithmMeta]:
        return Step.create_meta(self.algorithm_type)

    def identifier(self) -> Tuple:
        return ()

    @abstractmethod
    def run(self, task: Task, com_queue: Queue) -> dict:
        return {}

    def list_available_models_for_book(self, book: DatabaseBook) -> DatabaseAvailableModels:
        from database.models.bookstyles import BookStyle
        meta = self.algorithm_meta()
        return DatabaseAvailableModels(
            book=book.book,
            book_meta=book.get_meta(),
            newest_model=meta.newest_model_for_book(book).meta() if meta.newest_model_for_book(book) else None,
            selected_model=meta.selected_model_for_book(book).meta() if meta.selected_model_for_book(book) else None,
            book_models=[m.meta() for m in meta.models_for_book(book).list_models()],
            default_book_style_model=meta.model_of_book_style(book).meta() if meta.model_of_book_style(book) else None,
            models_of_same_book_style=[(b.get_meta(), meta.newest_model_for_book(b).meta()) for b in DatabaseBook.list_available_of_style(book.get_meta().notationStyle) if b.book != book.book and meta.newest_model_for_book(b)],
            default_models=[DefaultModel(o.id, meta.default_model_for_style(o.id).meta()) for o in BookStyle.objects.all() if meta.default_model_for_style(o.id)],
        )
