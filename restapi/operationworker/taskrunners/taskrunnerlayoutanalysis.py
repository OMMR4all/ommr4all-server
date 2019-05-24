from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes


class TaskRunnerLayoutAnalysis(TaskRunner):
    def __init__(self,
                 page: DatabasePage
                 ):
        super().__init__({TaskWorkerGroup.NORMAL_TASKS_CPU})
        self.page = page

    def identifier(self) -> Tuple:
        return (self.page, )

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.layout.predictor import LayoutPredictorParameters, create_predictor, PredictorTypes, \
            LayoutAnalysisPredictorCallback
        from database.file_formats import PcGts

        class Callback(LayoutAnalysisPredictorCallback):
            def __init__(self):
                super().__init__()

            def progress_updated(self, percentage: float):
                print(percentage)

                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.WORKING,
                    progress=percentage,
                )))

        params = LayoutPredictorParameters(checkpoints=[])
        pred = create_predictor(PredictorTypes.STANDARD, params)
        pcgts = PcGts.from_file(self.page.file('pcgts'))

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
        return list(pred.predict([pcgts], Callback()))[0].to_dict()
