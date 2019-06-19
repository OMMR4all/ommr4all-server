from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from .pageselection import PageSelection, DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
import logging
from typing import NamedTuple
from database.file_formats.pcgts import TextRegion, TextLine, MusicRegion, MusicLine, MusicLines

logger = logging.getLogger(__name__)


def unprocessed(page: DatabasePage) -> bool:
    return len(page.pcgts().page.text_regions) == 0


class Settings(NamedTuple):
    store_to_pcgts: bool = False


class TaskRunnerLayoutAnalysis(TaskRunner):
    def __init__(self,
                 selection: PageSelection,
                 settings: Settings,
                 ):
        super().__init__({TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.selection = selection
        self.settings = settings

    def identifier(self) -> Tuple:
        return self.selection.identifier(),

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.layout.predictor import LayoutPredictorParameters, create_predictor, PredictorTypes, \
            LayoutAnalysisPredictorCallback

        class Callback(LayoutAnalysisPredictorCallback):
            def __init__(self):
                super().__init__()

            def progress_updated(self, percentage: float):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.WORKING,
                    progress=percentage,
                )))

        params = LayoutPredictorParameters(checkpoints=[])
        pred = create_predictor(PredictorTypes.STANDARD, params)

        selected_pages = self.selection.get(unprocessed)
        pages = [p.pcgts() for p in selected_pages]

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
        logger.debug("Starting layout prediction")
        results = list(pred.predict(pages, Callback()))
        logger.debug("Finished layout prediction")
        if self.settings.store_to_pcgts:
            for page_layouts, pcgts, page in zip(results, pages, selected_pages):
                pcgts.page.text_regions.clear()
                for type, id_coords in page_layouts.text_regions.items():
                    for ic in id_coords:
                        pcgts.page.text_regions.append(
                            TextRegion(ic.id, type, None, [TextLine(coords=ic.coords)])
                        )

                for ic in page_layouts.music_regions:
                    ml = pcgts.page.music_line_by_id(ic.id)
                    if not ml:
                        logger.warning('Music line with id "{}" not found'.format(ic.id))
                        continue

                    ml.coords = ic.coords

                pcgts.to_file(page.file('pcgts').local_path())

        if self.selection.single_page:
            return results[0].to_dict()
        else:
            return {
                'pages': [r.to_dict() for r in results]
            }
