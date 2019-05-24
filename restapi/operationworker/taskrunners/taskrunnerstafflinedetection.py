from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage
import os
from django.conf import settings
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes


class TaskRunnerStaffLineDetection(TaskRunner):
    def __init__(self,
                 page: DatabasePage
                 ):
        super().__init__({TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.page = page

    def identifier(self) -> Tuple:
        return self.page,

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.stafflines.detection.predictor import \
            create_staff_line_predictor, StaffLinesModelType, StaffLinesPredictor, \
            StaffLinePredictorParameters, StaffLineDetectionDatasetParams, LineDetectionPredictorCallback
        import omr.stafflines.detection.pixelclassifier.settings as pc_settings
        from database.file_formats import PcGts
        # load book specific model or default model as fallback
        model = self.page.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))
        if not os.path.exists(model + '.meta'):
            model = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_models', 'french14', pc_settings.model_dir, pc_settings.model_name)

        class Callback(LineDetectionPredictorCallback):
            def __init__(self):
                super().__init__()

            def progress_updated(self, percentage: float):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.WORKING,
                    progress=percentage,
                )))

        params = StaffLinePredictorParameters(
            checkpoints=[model],
        )
        staff_line_detector: StaffLinesPredictor = create_staff_line_predictor(StaffLinesModelType.PIXEL_CLASSIFIER, params)
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
        staffs = list(staff_line_detector.predict([PcGts.from_file(self.page.file('pcgts'))], Callback()))[0].music_lines
        return {'staffs': [l.to_json() for l in staffs]}
