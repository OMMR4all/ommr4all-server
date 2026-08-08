from absl.logging import exception

from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, AlgorithmTypes
from ..workerresources import groups_for, WorkerResource
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .pageselection import PageSelection, DatabasePage
from typing import NamedTuple
import logging


logger = logging.getLogger(__name__)



class Settings(NamedTuple):
    params: AlgorithmPredictorParams
    store_to_pcgts: bool = False


class TaskRunnerPrediction(TaskRunner):
    def __init__(self,
                 algorithm_type: AlgorithmTypes,
                 selection: PageSelection,
                 settings: Settings,
                 worker_resource: WorkerResource = WorkerResource.CPU,
                 ):
        super().__init__(algorithm_type, selection, groups_for(worker_resource, training=False))
        self.settings = settings
        self.worker_resource = worker_resource
    def identifier(self) -> Tuple:
        return self.selection.identifier(), self.algorithm_type

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.steps.algorithm import PredictionCallback, AlgorithmPredictor, AlgorithmPredictorSettings
        from omr.steps import predictorcache
        meta = self.algorithm_meta()

        class Callback(PredictionCallback):
            def __init__(self):
                super().__init__()

            def progress_updated(self,
                                 percentage: float,
                                 n_pages: int = 0,
                                 n_processed_pages: int = 0,
                                 ):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.WORKING,
                    progress=percentage,
                    n_total=n_pages,
                    n_processed=n_processed_pages,
                )))

        params = AlgorithmPredictorSettings(
            model=meta.selected_model_for_book(self.selection.book),
            params=self.settings.params,
        )

        abc_detector: AlgorithmPredictor = predictorcache.get_or_create(meta, params)
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

        predictor_cls = meta.predictor()
        pages = self.selection.get_pages(predictor_cls.unprocessed, predictor_cls.unlocked)
        if not self.selection.single_page:
            # A batch run (workflow or book operation) must never overwrite a page the user
            # locked for this step, no matter which count mode was selected. Single-page
            # runs come from the editor, where re-running on a locked page is deliberate.
            pages = [p for p in pages if predictor_cls.unlocked(p)]
        logger.debug("Algorithm {} processing {} pages".format(self.algorithm_type.name, len(pages)))

        staves = list(abc_detector.predict(pages, Callback()))
        results = [
            page_staves.to_dict() for page_staves in staves
        ]

        if self.settings.store_to_pcgts:
            for page_staves in staves:
                page_staves.store_to_page()
            if len(staves) > 0:
                for page in pages:
                    page.mark_updated(task.creator, propagate=False)
                self.selection.book.mark_updated(task.creator)

        if self.selection.single_page:
            return results[0]
        else:
            return {
                'results': results
            }
