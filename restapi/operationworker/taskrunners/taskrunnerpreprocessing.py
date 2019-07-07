from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, AlgorithmTypes
from database.database_page import DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from restapi.operationworker.taskrunners.pageselection import PageSelection
from omr.steps.preprocessing.predictor import files, PredictionCallback, AlgorithmPredictorParams, AlgorithmPredictorSettings
import logging


logger = logging.getLogger(__name__)


class TaskRunnerPreprocessing(TaskRunner):
    def __init__(self,
                 selection: PageSelection,
                 params: AlgorithmPredictorParams,
                 ):
        super().__init__(AlgorithmTypes.PREPROCESSING, {TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.selection = selection
        self.params = params

    def identifier(self) -> Tuple:
        return self.selection.identifier()

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return any([not page.file(f).exists() for f in files])

    def run(self, task: Task, com_queue: Queue) -> dict:
        pages = self.selection.get_pages(TaskRunnerPreprocessing.unprocessed)
        logger.debug("Starting preprocessing of {} pages".format(len(pages)))

        class Callback(PredictionCallback):
            def progress_updated(self,
                                 percentage: float,
                                 n_pages: int = 0,
                                 n_processed_pages: int = 0):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.WORKING,
                    progress=percentage,
                    n_processed=n_processed_pages,
                    n_total=n_pages,
                )))

        cb = Callback()

        meta = self.algorithm_meta()
        settings = AlgorithmPredictorSettings(
            model=meta.best_model_for_book(self.selection.book),
            params=self.params,
        )
        pred = meta.create_predictor(settings)
        pred.predict(pages, cb)

        logger.debug("Finished preprocessing of {} pages".format(len(pages)))
        return {}

