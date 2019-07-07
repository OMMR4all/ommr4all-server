from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, AlgorithmTypes
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .pageselection import PageSelection, DatabasePage
from typing import NamedTuple
from database.file_formats.pcgts import Block, BlockType


class Settings(NamedTuple):
    params: AlgorithmPredictorParams
    store_to_pcgts: bool = False


class TaskRunnerStaffLineDetection(TaskRunner):
    def __init__(self,
                 selection: PageSelection,
                 settings: Settings,
                 ):
        super().__init__(AlgorithmTypes.STAFF_LINES_PC, {TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.selection = selection
        self.settings = settings

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return len(page.pcgts().page.music_blocks()) == 0

    def identifier(self) -> Tuple:
        return self.selection.identifier(),

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.steps.algorithm import PredictionCallback, AlgorithmPredictor, AlgorithmPredictorSettings
        meta = self.algorithm_meta()

        class Callback(PredictionCallback):
            def __init__(self, n_total):
                super().__init__()
                self.n_total = n_total

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
            model=meta.selected_model_for_book(self.selection.book)
        )
        staff_line_detector: AlgorithmPredictor = meta.create_predictor(params)
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

        pages = self.selection.get_pcgts(TaskRunnerStaffLineDetection.unprocessed)
        selected_pages = [p.page.location for p in pages]

        staves = list(staff_line_detector.predict(pages, Callback(len(pages))))
        results = [
            {
                'staffs': [l.to_json() for l in page_staves.music_lines],
                'page': page.page,
                'book': page.book.book,
            } for page_staves, page in zip(staves, selected_pages)
        ]

        if self.settings.store_to_pcgts:
            for page_staves, pcgts, page in zip(staves, pages, selected_pages):
                pcgts.page.clear_blocks_of_type(BlockType.MUSIC)
                for ml in page_staves.music_lines:
                    pcgts.page.blocks.append(
                        Block(BlockType.MUSIC, lines=[ml])
                    )

                pcgts.to_file(page.file('pcgts').local_path())

        if self.selection.single_page:
            return results[0]
        else:
            return {
                'pages': results
            }
