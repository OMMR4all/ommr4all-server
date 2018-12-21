from django.conf import settings
import threading
from multiprocessing import Process, Lock, Queue
import time
from typing import NamedTuple, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from queue import Empty as QueueEmptyException
import os

from main.book import Page, Book

import logging
logger = logging.getLogger(__name__)


class TaskNotFoundException(Exception):
    pass


class TaskNotFinishedException(Exception):
    pass


class TaskStatusCodes(Enum):
    QUEUED = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3
    NOT_FOUND = 4


class TaskProgressCodes(Enum):
    INITIALIZING = 0
    WORKING = 1
    FINALIZING = 2


@dataclass
class TaskStatus:
    code: TaskStatusCodes = TaskStatusCodes.NOT_FOUND
    progress_code: TaskProgressCodes = TaskProgressCodes.INITIALIZING
    progress: float = -1
    accuracy: float = -1
    early_stopping_progress: float = -1
    loss: float = -1

    def to_json(self):
        return {
            'code': self.code.value,
            'progress_code': self.progress_code.value,
            'progress': self.progress,
            'accuracy': self.accuracy,
            'early_stopping_progress': self.early_stopping_progress,
            'loss': self.loss,
        }


@dataclass
class Task:
    task_data: Any
    task_status: TaskStatus
    task_result: dict


class TaskQueue:
    def __init__(self):
        self.tasks: List[Task] = []
        self.mutex = Lock()

    def remove(self, task_data) -> Optional[Task]:
        with self.mutex:
            for i, t in enumerate(self.tasks):
                if t.task_data == task_data:
                    del self.tasks[i]
                    return t

            return None

    def put(self, task_data) -> bool:
        with self.mutex:
            for task in self.tasks:
                if task.task_data == task_data:
                    return False

            self.tasks.append(Task(task_data, TaskStatus(code=TaskStatusCodes.QUEUED), {}))

            return True

    def pop_result(self, task_data) -> dict:
        with self.mutex:
            for i, t in enumerate(self.tasks):
                if t.task_data == task_data:
                    if t.task_status.code == TaskStatusCodes.QUEUED or t.task_status.code == TaskStatusCodes.RUNNING:
                        raise TaskNotFinishedException()

                    del self.tasks[i]
                    return t.task_result

            raise TaskNotFoundException()

    def status(self, task_data) -> TaskStatus:
        with self.mutex:
            for task in self.tasks:
                if task.task_data == task_data:
                    return task.task_status

            return TaskStatus(TaskStatusCodes.NOT_FOUND)

    def update_status(self, task_data, status: TaskStatus):
        with self.mutex:
            for task in self.tasks:
                if task.task_data == task_data:
                    task.task_status = status
                    return

            raise TaskNotFoundException()

    def next_unprocessed(self, sleep_secs=1.0) -> Task:
        while True:
            with self.mutex:
                for task in self.tasks:
                    if task.task_status.code == TaskStatusCodes.QUEUED:
                        task.task_status.code = TaskStatusCodes.RUNNING
                        return task

            time.sleep(sleep_secs)

    def task_finished(self, task, result):
        with self.mutex:
            task.task_status.code = TaskStatusCodes.FINISHED
            task.task_result = result

    def task_error(self, task, result):
        with self.mutex:
            task.task_status.code = TaskStatusCodes.ERROR
            task.task_result = result


class TaskDataStaffLineDetection(NamedTuple):
    page: Page


class TaskDataSymbolDetection(NamedTuple):
    page: Page


class TaskDataSymbolDetectionTrainer(NamedTuple):
    book: Book


class OperationWorkerThread:
    def __init__(self, thread_id, queue: TaskQueue, com_queue: Queue):
        self.com_queue = com_queue
        self.thread = threading.Thread(target=self.run, args=(), name=thread_id)
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.sleep_interval = 1.0  # seconds
        self.queue = queue
        self._current_task = None
        self.mutex = Lock()
        self.process = None

        self.thread.start()

    def cancel_if_current_task(self, task) -> bool:
        if self._current_task is None:
            return False

        with self.mutex:
            if self._current_task == task and self.process:
                logger.info('THREAD {}: Attempting to terminate thread'.format(self.thread.name))
                self.process.terminate()
                self.process.join()
                logger.info('THREAD {}: Thread terminated'.format(self.thread.name))
                return True

            return False

    def run(self):
        logger.info('THREAD {}: Thread started'.format(self.thread.name))
        while True:
            logger.debug('THREAD {}: Waiting for new Task'.format(self.thread.name))
            try:
                with self.mutex:
                    self._current_task = self.queue.next_unprocessed(self.sleep_interval)
                    logger.info('THREAD {}: Running new task of type {}'.format(self.thread.name, type(self._current_task.task_data)))

                    queue = Queue()
                    self.process = Process(target=OperationWorkerThread._run_task, args=(self.thread.name, self._current_task, queue, self.com_queue))
                    self.process.daemon = False     # must be stopped explicitly
                    self.process.start()

                self.process.join()                 # wait for the task to finish
                result = queue.get(timeout=0)       # get data NOW, or throw a QueueEmptyException (if process canceled)

                if self.process is None:
                    # process canceled
                    raise TaskNotFinishedException()

                if isinstance(result, Exception):
                    logger.info("THREAD {}: Exception during Task-Execution: {}".format(self.thread.name, result))
                    raise result
            except (QueueEmptyException, BrokenPipeError, TaskNotFinishedException) as e:
                with self.mutex:
                    self.queue.task_error(self._current_task, e)
                    self._current_task = None
                logger.info("THREAD {}: Task canceled".format(self.thread.name))
            except Exception as e:
                with self.mutex:
                    self.queue.task_error(self._current_task, e)
                    self._current_task = None
                logger.exception("THREAD {}: Error in thread: {}".format(self.thread.name, e))
            else:  # Successfully finished!
                logger.debug('THREAD {}: Task finished successfully'.format(self.thread.name))
                self.queue.task_finished(self._current_task, result)
            finally:  # Always clear current data
                with self.mutex:
                    self.process = None
                    self._current_task = None

    @staticmethod
    def _run_task(name: str, task: Task, queue: Queue, com_queue: Queue):
        try:
            start = time.time()
            task_data = task.task_data
            result = {}
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.INITIALIZING)))
            if isinstance(task_data, TaskDataStaffLineDetection):
                data: TaskDataStaffLineDetection = task_data
                from omr.stafflines.detection.staffline_detector import create_staff_line_detector, StaffLineDetectorType, StaffLineDetector
                staff_line_detector: StaffLineDetector = create_staff_line_detector(StaffLineDetectorType.BASIC, data.page)
                com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
                staffs = staff_line_detector.detect(
                    data.page.file('binary_deskewed').local_path(),
                    data.page.file('gray_deskewed').local_path(),
                )
                result = {'staffs': [l.to_json() for l in staffs]}
            elif isinstance(task_data, TaskDataSymbolDetection):
                data: TaskDataSymbolDetection = task_data
                from omr.symboldetection.predictor import PredictorParameters, PredictorTypes, create_predictor
                import omr.symboldetection.pixelclassifier.settings as pc_settings
                from omr.datatypes import PcGts

                # load book specific model or default model as fallback
                model = data.page.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))
                if not os.path.exists(model + '.meta'):
                    model = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_models', 'french14', pc_settings.model_dir, pc_settings.model_name)

                params = PredictorParameters(
                    checkpoints=[model],
                )
                pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER, params)
                pcgts = PcGts.from_file(data.page.file('pcgts'))

                com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))

                music_lines = []
                for i, line_prediction in enumerate(pred.predict([pcgts])):
                    com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING)))
                    music_lines.append({'symbols': [s.to_json() for s in line_prediction.symbols],
                                        'id': line_prediction.line.operation.music_line.id})
                result = {'musicLines': music_lines}
            elif isinstance(task_data, TaskDataSymbolDetectionTrainer):
                data: TaskDataSymbolDetectionTrainer = task_data
                from omr.symboldetection.trainer import SymbolDetectionTrainer, SymbolDetectionTrainerCallback

                class Callback(SymbolDetectionTrainerCallback):
                    def __init__(self):
                        super().__init__()
                        self.iter, self.loss, self.acc, self.best_iter, self.best_acc, self.best_iters = -1, -1, -1, -1, -1, -1

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

                SymbolDetectionTrainer(data.book, callback=Callback())
            else:
                logger.exception('THREAD {}: Unknown operation data {} of task {}'.format(name, task_data, task))

            logger.info("THREAD {}: Task finished. It ran for {}s".format(name, time.time() - start))
            queue.put(result)
        except Exception as e:
            logger.exception("THREAD {}: Error in thread: {}".format(name, e))
            queue.put(e)


class TaskCommunicationData(NamedTuple):
    task: Task
    status: TaskStatus


class TaskCommunicator:
    def __init__(self, task_queue: TaskQueue):
        self.task_queue: TaskQueue = task_queue
        self.queue = Queue()
        self.thread = threading.Thread(target=self.run, args=(), name='task_communicator')
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.thread.start()

    def run(self):
        logger.info("THREAD task_communicator: Started")
        while True:
            try:
                com: TaskCommunicationData = self.queue.get()
                self.task_queue.update_status(com.task.task_data, com.status)
            except TaskNotFoundException:
                pass
            except Exception as e:
                raise e


class OperationWorker:
    def __init__(self, num_threads=1):
        self.queue = TaskQueue()
        self.task_communicator = TaskCommunicator(self.queue)
        self.threads = [OperationWorkerThread(i, self.queue, self.task_communicator.queue) for i in range(num_threads)]

    def stop(self, task_data):
        task = self.queue.remove(task_data)
        if task is not None:
            for i, t in enumerate(self.threads):
                if t.cancel_if_current_task(task):
                    break

    def put(self, task_data) -> bool:
        return self.queue.put(task_data)

    def pop_result(self, task_data) -> dict:
        return self.queue.pop_result(task_data)

    def status(self, task_data) -> Optional[TaskStatus]:
        return self.queue.status(task_data)


operation_worker = OperationWorker()


if __name__ == '__main__':
    print('done')
