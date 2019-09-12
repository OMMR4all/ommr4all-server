from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple, AlgorithmTypes, PageSelection
from database import DatabaseBook, DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .trainerparams import TaskTrainerParams
import logging
from omr.dataset.datafiles import dataset_by_locked_pages, LockState
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerSettings, DatasetParams
from omr.dataset.dataset import PageScaleReference

logger = logging.getLogger(__name__)


class TaskRunnerSymbolDetectionTrainer(TaskRunner):
    def __init__(self,
                 book: DatabaseBook,
                 params: TaskTrainerParams,
                 ):
        super().__init__(AlgorithmTypes.SYMBOLS_PC,
                         PageSelection.from_book(book),
                         [TaskWorkerGroup.LONG_TASKS_GPU, TaskWorkerGroup.LONG_TASKS_CPU])
        self.params = params

    def identifier(self) -> Tuple:
        return self.selection.identifier(),

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return True

    def run(self, task: Task, com_queue: Queue) -> dict:
        class Callback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.iter, self.loss, self.acc, self.best_iter, self.best_acc, self.best_iters = -1, -1, -1, -1, -1, -1

            def resolving_files(self):
                com_queue.put(TaskCommunicationData(task, TaskStatus(
                    TaskStatusCodes.RUNNING,
                    TaskProgressCodes.RESOLVING_DATA,
                )))

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

        callback = Callback()

        logger.info("Finding PcGts files with valid ground truth")
        callback.resolving_files()
        train_pcgts, val_pcgts = dataset_by_locked_pages(
            self.params.nTrain, [LockState('Symbols', True)],
            datasets=[self.selection.book] if not self.params.includeAllTrainingData else [])
        if len(train_pcgts) + len(val_pcgts) < 50:
            # only very few files, use all for training and evaluate on training as-well
            train_pcgts = train_pcgts + val_pcgts
            val_pcgts = train_pcgts
            logger.info("Combining training and validation files because n<50")

        logger.info("Starting training with {} training and {} validation files".format(len(train_pcgts), len(val_pcgts)))
        logger.debug("Training files: {}".format([p.page.location.local_path() for p in train_pcgts]))
        logger.debug("Validation files: {}".format([p.page.location.local_path() for p in val_pcgts]))

        meta = self.algorithm_meta()
        train, val = self.params.to_train_val(locks=[LockState('StaffLines', True)], books=[self.selection.book])

        settings = AlgorithmTrainerSettings(
            train_data=train,
            validation_data=val,
            dataset_params=DatasetParams(
                gt_required=True,
                pad=None,
                pad_power_of_2=3,
                height=80,
                dewarp=False,
                cut_region=False,
                center=True,
                staff_lines_only=True,
            ),
        )

        trainer = meta.create_trainer(settings)
        if self.params.pretrainedModel:
            trainer.settings.params.load = self.params.pretrainedModel.id
        trainer.train(self.selection.book, callback=callback)
        logger.info("Training finished for book {}".format(self.selection.book.local_path()))
        return {}
