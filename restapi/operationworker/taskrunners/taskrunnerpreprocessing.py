from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage, DatabaseBook
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from typing import Optional, List, NamedTuple
import multiprocessing
import logging


logger = logging.getLogger(__name__)


from enum import Enum

class PageCount(Enum):
    ALL = 'all'
    UNPROCESSED = 'unprocessed'
    CUSTOM = 'custom'


class JsonParseKeyNotFound(Exception):
    def __init__(self, key: str, d: dict):
        self.key = key
        self.d = d


files = ['color_norm', 'color_highres_preproc', 'color_lowres_preproc', 'connected_components_norm']

def require_json(d: dict, key: str):
    if not key in d:
        raise JsonParseKeyNotFound(key, d)

    return d[key]


class Settings(NamedTuple):
    average_line_distance: int
    automatic_line_distance: bool

    @staticmethod
    def from_json(json: dict):
        return Settings(
            json.get('avgLd', 10),
            require_json(json, 'automaticLd'),
        )


class PageSelection:
    def __init__(self,
                 book: DatabaseBook,
                 page_count: PageCount,
                 pages: Optional[List[DatabasePage]] = None,
                 ):
        self.book = book
        self.page_count = page_count
        self.pages = pages if pages else []

    @staticmethod
    def from_json(d: dict, book: DatabaseBook):
        if not 'count' in d:
            raise JsonParseKeyNotFound('count', d)

        return PageSelection(
            book,
            PageCount(require_json(d, 'count')),
            [book.page(page) for page in d.get('pages', [])]
        )

    def identifier(self) -> Tuple:
        return self.book, self.page_count, self.pages

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.identifier() == other.identifier()

    def get(self) -> List[DatabasePage]:
        if self.page_count == PageCount.ALL:
            return self.book.pages()
        elif self.page_count == PageCount.UNPROCESSED:
            return [p for p in self.book.pages() if any([not p.file(f).exists() for f in files])]
        else:
            return self.pages


def _process_single(args: Tuple[DatabasePage, Settings]):
    page, settings = args
    for file in files:
        # create or recreate files
        file = page.file(file)
        file.delete()
        file.create()


class TaskRunnerPreprocessing(TaskRunner):
    def __init__(self,
                 selection: PageSelection,
                 settings: Settings,
                 ):
        super().__init__({TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.selection = selection
        self.settings = settings

    def identifier(self) -> Tuple:
        return self.selection.identifier()

    def run(self, task: Task, com_queue: Queue) -> dict:
        pages = self.selection.get()
        com_queue.put(TaskCommunicationData(task, TaskStatus(
            TaskStatusCodes.RUNNING,
            TaskProgressCodes.WORKING,
            progress=0,
            n_processed=0,
            n_total=len(pages),
        )))
        pool = multiprocessing.Pool(processes=4)
        for i, _ in enumerate(pool.imap_unordered(_process_single, [(p, self.settings) for p in pages])):
            percentage = (i + 1) / len(pages)
            com_queue.put(TaskCommunicationData(task, TaskStatus(
                TaskStatusCodes.RUNNING,
                TaskProgressCodes.WORKING,
                progress=percentage,
                n_processed=i + 1,
                n_total=len(pages),
            )))

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

        logger.debug("Starting preprocessing of {} pages".format(len(pages)))
        logger.debug("Finished preprocessing of {} pages".format(len(pages)))
        return {}

