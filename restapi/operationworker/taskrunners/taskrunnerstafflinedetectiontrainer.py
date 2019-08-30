from database.file_formats.performance.pageprogress import Locks
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, DatabaseAvailableModels, AlgorithmTypes, PageSelection
from database import DatabaseBook, DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .trainerparams import TaskTrainerParams, LockState
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerSettings, DatasetParams
from omr.dataset.dataset import PageScaleReference


class TaskRunnerStaffLineDetectionTrainer(TaskRunner):
    def __init__(self,
                 book: DatabaseBook,
                 params: TaskTrainerParams,
                 ):
        super().__init__(AlgorithmTypes.STAFF_LINES_PC,
                         PageSelection.from_book(book),
                         [TaskWorkerGroup.LONG_TASKS_GPU, TaskWorkerGroup.LONG_TASKS_CPU])
        self.params = params

    def identifier(self) -> Tuple:
        return self.selection.identifier(),

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return True

    def run(self, task: Task, com_queue: Queue) -> dict:
        book = self.selection.book
        meta = self.algorithm_meta()

        class Callback(TrainerCallback):
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

        train, val = self.params.to_train_val(locks=[LockState(Locks.STAFF_LINES, True)], books=[book])

        settings = AlgorithmTrainerSettings(
            train_data=train,
            validation_data=val,
            dataset_params=DatasetParams(
                gt_required=True,
                pad=None,
                pad_power_of_2=3,
                full_page=True,
                gray=True,
                extract_region_only=True,
                gt_line_thickness=2,
                page_scale_reference=PageScaleReference.NORMALIZED,
                target_staff_line_distance=10,
                origin_staff_line_distance=10,
            ),
        )

        trainer = meta.create_trainer(settings)
        if self.params.pretrainedModel:
            trainer.settings.params.load = self.params.pretrainedModel.id
        trainer.train(book, callback=Callback())
        return {}
