from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
import os
from django.conf import settings
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .pageselection import PageSelection, DatabasePage
from typing import NamedTuple
from database.file_formats.pcgts import MusicRegion, MusicLines


class Settings(NamedTuple):
    store_to_pcgts: bool = False


class TaskRunnerStaffLineDetection(TaskRunner):
    def __init__(self,
                 selection: PageSelection,
                 settings: Settings,
                 ):
        super().__init__({TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.selection = selection
        self.settings = settings

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return len(page.pcgts().page.music_regions) == 0

    def identifier(self) -> Tuple:
        return self.selection.identifier(),

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.stafflines.detection.predictor import \
            create_staff_line_predictor, StaffLinesModelType, StaffLinesPredictor, \
            StaffLinePredictorParameters, StaffLineDetectionDatasetParams, LineDetectionPredictorCallback
        import omr.stafflines.detection.pixelclassifier.settings as pc_settings
        # load book specific model or default model as fallback
        model = self.selection.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))
        if not os.path.exists(model + '.meta'):
            model = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_models', 'french14', pc_settings.model_dir, pc_settings.model_name)

        class Callback(LineDetectionPredictorCallback):
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

        params = StaffLinePredictorParameters(
            checkpoints=[model],
        )
        staff_line_detector: StaffLinesPredictor = create_staff_line_predictor(StaffLinesModelType.PIXEL_CLASSIFIER, params)
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
        selected_pages = self.selection.get(TaskRunnerStaffLineDetection.unprocessed)
        pages = [p.pcgts() for p in selected_pages]
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
                pcgts.page.music_regions.clear()
                for ml in page_staves.music_lines:
                    pcgts.page.music_regions.append(
                        MusicRegion(staffs=MusicLines([ml]))
                    )

                pcgts.to_file(page.file('pcgts').local_path())

        if self.selection.single_page:
            return results[0]
        else:
            return {
                'pages': results
            }
