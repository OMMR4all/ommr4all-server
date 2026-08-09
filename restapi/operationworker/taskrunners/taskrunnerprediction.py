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
        from omr.steps.algorithm import PredictionCallback, AlgorithmPredictor, AlgorithmPredictorSettings, \
            FailedPageResult
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

        all_results = list(abc_detector.predict(pages, Callback()))
        # A predictor that guards its per-page loop reports a failed page instead of
        # aborting the run. Those pages are kept out of the results and never stored,
        # but are reported back so the client can warn about them.
        failed = [r for r in all_results if isinstance(r, FailedPageResult)]
        staves = [r for r in all_results if not isinstance(r, FailedPageResult)]
        results = [
            page_staves.to_dict() for page_staves in staves
        ]

        if failed:
            logger.warning("Algorithm {} skipped {} of {} pages: {}".format(
                self.algorithm_type.name, len(failed), len(pages),
                ", ".join("{} ({})".format(f.page_name, f.error) for f in failed)))

        if self.settings.store_to_pcgts:
            failed_pages = {f.page_name for f in failed}
            for page_staves in staves:
                page_staves.store_to_page()
            if len(staves) > 0:
                for page in pages:
                    if page.page in failed_pages:
                        continue
                    page.mark_updated(task.creator, propagate=False)
                self.selection.book.mark_updated(task.creator)

        if self.selection.single_page:
            if len(results) == 0:
                raise Exception("Prediction of {} produced no result for page {}{}".format(
                    self.algorithm_type.name,
                    pages[0].page if pages else '<none>',
                    ": " + failed[0].error if failed else ''))
            return results[0]
        else:
            return {
                'results': results,
                'skipped_pages': [f.to_dict() for f in failed],
            }
