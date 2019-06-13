from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage
import os
from django.conf import settings
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes


class TaskRunnerSymbolDetection(TaskRunner):
    def __init__(self,
                 page: DatabasePage
                 ):
        super().__init__({TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.page = page

    def identifier(self) -> Tuple:
        return self.page,

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.symboldetection.predictor import \
            SymbolDetectionPredictorParameters, PredictorTypes, create_predictor, SymbolDetectionDatasetParams
        import omr.symboldetection.pixelclassifier.settings as pc_settings
        from database.file_formats import PcGts

        # load book specific model or default model as fallback
        model = self.page.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))
        if not os.path.exists(model + '.meta'):
            model = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_models', 'french14', pc_settings.model_dir, pc_settings.model_name)

        params = SymbolDetectionPredictorParameters(
            checkpoints=[model],
            symbol_detection_params=SymbolDetectionDatasetParams()
        )
        pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER, params)
        pcgts = PcGts.from_file(self.page.file('pcgts'))

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

        music_lines = []
        for i, line_prediction in enumerate(pred.predict([pcgts])):
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
            music_lines.append({'symbols': [s.to_json() for s in line_prediction.symbols],
                                'id': line_prediction.line.operation.music_line.id})
        return {'musicLines': music_lines}
