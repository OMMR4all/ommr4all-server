from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from .pageselection import PageSelection, DatabasePage
import os
from django.conf import settings
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from typing import NamedTuple


def unprocessed(page: DatabasePage) -> bool:
    return all([len(l.symbols) == 0 for l in page.pcgts().page.all_music_lines()])


class Settings(NamedTuple):
    store_to_pcgts: bool = False


class TaskRunnerSymbolDetection(TaskRunner):
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
        from omr.symboldetection.predictor import \
            SymbolDetectionPredictorParameters, PredictorTypes, create_predictor, SymbolDetectionDatasetParams
        import omr.symboldetection.pixelclassifier.settings as pc_settings

        selected_pages = self.selection.get(unprocessed)
        pages = [p.pcgts() for p in selected_pages]

        # load book specific model or default model as fallback
        model = self.selection.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))
        if not os.path.exists(model + '.meta'):
            model = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_models', 'french14', pc_settings.model_dir, pc_settings.model_name)

        params = SymbolDetectionPredictorParameters(
            checkpoints=[model],
            symbol_detection_params=SymbolDetectionDatasetParams()
        )
        pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER, params)

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
                pcgts.to_file(page.file('pcgts').local_path())

        return {'musicLines': music_lines}
