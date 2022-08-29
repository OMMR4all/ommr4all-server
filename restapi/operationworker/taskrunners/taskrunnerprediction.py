from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, AlgorithmTypes
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
                 ):
        super().__init__(algorithm_type, selection, [TaskWorkerGroup.NORMAL_TASKS_CPU])
        self.settings = settings
    def identifier(self) -> Tuple:
        return self.selection.identifier(), self.algorithm_type

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.steps.algorithm import PredictionCallback, AlgorithmPredictor, AlgorithmPredictorSettings
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
        staff_line_detector: AlgorithmPredictor = meta.create_predictor(params)
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

        pages = self.selection.get_pages(meta.predictor().unprocessed)
        logger.debug("Algorithm {} processing {} pages".format(self.algorithm_type.name, len(pages)))

        staves = list(staff_line_detector.predict(pages, Callback()))
        results = [
            page_staves.to_dict() for page_staves in staves
        ]
        if self.settings.store_to_pcgts:
            for page_staves in staves:
                page_staves.store_to_page()

        if self.selection.single_page:
            return results[0]
        else:
            return {
                'results': results
            }
