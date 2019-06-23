from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from .pageselection import PageSelection, DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
import logging
from typing import NamedTuple
from enum import Enum
from database.file_formats.pcgts import BlockType, Block, Line
from omr.layout.predictor import PredictorTypes


logger = logging.getLogger(__name__)


class LayoutModes(Enum):
    SIMPLE = 'simple'
    COMPLEX = 'complex'

    def to_predictor_type(self) -> PredictorTypes:
        return {
            LayoutModes.SIMPLE: PredictorTypes.LYRICS_BBS,
            LayoutModes.COMPLEX: PredictorTypes.STANDARD,
        }[self]


class Settings(NamedTuple):
    store_to_pcgts: bool = False
    layout_mode: LayoutModes = LayoutModes.COMPLEX

    @staticmethod
    def from_json(d: dict):
        return Settings(
            d.get('storeToPcGts', False),
            LayoutModes(d.get('layoutMode', LayoutModes.COMPLEX.value)),
        )


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

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return len(page.pcgts().page.text_blocks()) == 0

    def run(self, task: Task, com_queue: Queue) -> dict:
        logger.debug("Starting layout prediction with mode {}".format(self.settings.layout_mode))

        from omr.layout.predictor import LayoutPredictorParameters, create_predictor, \
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
        pred = create_predictor(self.settings.layout_mode.to_predictor_type(), params)

        selected_pages = self.selection.get(TaskRunnerLayoutAnalysis.unprocessed)
        pages = [p.pcgts() for p in selected_pages]

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
        results = list(pred.predict(pages, Callback()))
        logger.debug("Finished layout prediction")
        if self.settings.store_to_pcgts:
            for page_layouts, pcgts, page in zip(results, pages, selected_pages):
                pcgts.page.clear_text_blocks()
                for type, id_coords in page_layouts.blocks.items():
                    for ic in id_coords:
                        if type == BlockType.MUSIC:
                            ml = pcgts.page.music_line_by_id(ic.id)
                            if not ml:
                                logger.warning('Music line with id "{}" not found'.format(ic.id))
                                continue

                            ml.coords = ic.coords
                        else:
                            pcgts.page.blocks.append(
                                Block(type, ic.id, lines=[Line(coords=ic.coords)])
                            )

                pcgts.to_file(page.file('pcgts').local_path())

        if self.selection.single_page:
            return results[0].to_dict()
        else:
            return {
                'pages': [r.to_dict() for r in results]
            }
