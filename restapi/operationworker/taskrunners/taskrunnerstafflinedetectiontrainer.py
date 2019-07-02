from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database import DatabaseBook, DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .trainerparams import TaskTrainerParams


class TaskRunnerStaffLineDetectionTrainer(TaskRunner):
    def __init__(self,
                 book: DatabaseBook,
                 params: TaskTrainerParams,
                 ):
        super().__init__({TaskWorkerGroup.LONG_TASKS_GPU})
        self.book = book
        self.params = params

    def identifier(self) -> Tuple:
        return self.book,

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return True

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.stafflines.detection.pixelclassifier.trainer import BasicStaffLinesTrainer, StaffLinesDetectionTrainerCallback

        class Callback(StaffLinesDetectionTrainerCallback):
            def __init__(self):
                super().__init__()
                self.iter, self.loss, self.acc, self.best_iter, self.best_acc, self.best_iters = -1, -1, -1, -1, -1, -1

            def loading(self, n: int, total: int):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.LOADING_DATA,
                    progress=n / total,
                    n_processed=n,
                    n_total=total,
                )))

            def loading_started(self, total: int):
                pass

            def loading_finished(self, total: int):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.PREPARING_TRAINING,
                )))

            def put(self):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.WORKING,
                    progress=self.iter / self.total_iters,
                    accuracy=self.best_acc if self.best_acc >= 0 else -1,
                    early_stopping_progress=self.best_iters / self.early_stopping_iters if self.early_stopping_iters > 0 else -1,
                    loss=self.loss,
                )))

            def next_iteration(self, iter: int, loss: float, acc: float):
                self.iter, self.loss, self.acc = iter, loss, acc
                self.put()

            def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
                self.best_iter, self.best_acc, self.best_iters = best_iter, best_acc, best_iters
                self.put()

            def early_stopping(self):
                pass

            def resolving_files(self):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.RESOLVING_DATA,
                )))

        trainer = BasicStaffLinesTrainer(self.book, self.params)
        trainer.train(callback=Callback())
        return {}
