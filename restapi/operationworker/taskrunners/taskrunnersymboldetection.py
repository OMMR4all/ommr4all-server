from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, AlgorithmTypes
from .pageselection import PageSelection, DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from typing import NamedTuple
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from database.model import Model


class Settings(NamedTuple):
    params: AlgorithmPredictorParams
    store_to_pcgts: bool = False


class TaskRunnerSymbolDetection(TaskRunner):
    def __init__(self,
                 selection: PageSelection,
                 settings: Settings,
                 ):
        super().__init__(AlgorithmTypes.SYMBOLS_PC, {TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.selection = selection
        self.settings = settings

    def identifier(self) -> Tuple:
        return self.selection.identifier(),

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return all([len(l.symbols) == 0 for l in page.pcgts().page.all_music_lines()])

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.steps.algorithm import AlgorithmPredictorSettings

        pages = self.selection.get_pcgts(TaskRunnerSymbolDetection.unprocessed)
        selected_pages = [p.page.location for p in pages]

        # load book specific model or default model as fallback
        meta = self.algorithm_meta()
        params = AlgorithmPredictorSettings(
            model=Model.from_id(self.settings.params.modelId) if self.settings.params.modelId else meta.selected_model_for_book(self.selection.book)
        )
        pred = meta.create_predictor(params)

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

        music_lines = []
        for i, line_prediction in enumerate(pred.predict(pages)):
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
            music_lines.append({'symbols': [s.to_json() for s in line_prediction.symbols],
                                'id': line_prediction.line.operation.music_line.id})

            if self.settings.store_to_pcgts:
                line_prediction.line.operation.music_line.symbols = line_prediction.symbols

        if self.settings.store_to_pcgts:
            for pcgts, page in zip(pages, selected_pages):
                pcgts.page.annotations.connections.clear()
                pcgts.to_file(page.file('pcgts').local_path())

        return {'musicLines': music_lines}
