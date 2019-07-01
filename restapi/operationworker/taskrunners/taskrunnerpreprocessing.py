from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from typing import NamedTuple
from restapi.operationworker.taskrunners.pageselection import PageSelection, require_json
import multiprocessing
import logging


logger = logging.getLogger(__name__)


files = ['color_norm', 'color_norm_x2', 'color_highres_preproc', 'color_lowres_preproc', 'connected_components_norm']


class Settings(NamedTuple):
    average_line_distance: int
    automatic_line_distance: bool

    @staticmethod
    def from_json(json: dict):
        return Settings(
            json.get('avgLd', 10),
            require_json(json, 'automaticLd'),
        )


def _process_single(args: Tuple[DatabasePage, Settings]):
    page, settings = args

    # update page meta
    meta = page.meta()
    meta.preprocessing.average_line_distance = settings.average_line_distance
    meta.preprocessing.auto_line_distance = settings.automatic_line_distance
    meta.save(page)

    # process all files
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

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return any([not page.file(f).exists() for f in files])

    def run(self, task: Task, com_queue: Queue) -> dict:
        pages = self.selection.get(TaskRunnerPreprocessing.unprocessed)
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

