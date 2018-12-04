from django.conf import settings
import threading
from multiprocessing import Process, Lock, Queue
import time
from typing import NamedTuple, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from queue import Empty as QueueEmptyException

from main.book import Page

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


@dataclass
class TaskStatus:
    code: TaskStatusCodes = TaskStatusCodes.QUEUED
    progress: float = 0
    accuracy: float = 0

    def to_json(self):
        return {'code': self.code.value,
                'progress': self.progress,
                'accuracy': self.accuracy,
                }


@dataclass
class Task:
    task_data: Any
    task_status: TaskStatus
    task_result: Any


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

            self.tasks.append(Task(task_data, TaskStatus(), None))

            return True

    def pop_result(self, task_data) -> Any:
        with self.mutex:
            for i, t in enumerate(self.tasks):
                if t.task_data == task_data:
                    if t.task_status.code == TaskStatusCodes.QUEUED or t.task_status.code == TaskStatusCodes.RUNNING:
                        raise TaskNotFinishedException()

                    del self.tasks[i]
                    return t.task_result

            raise TaskNotFoundException()

    def status(self, task_data) -> Optional[TaskStatus]:
        with self.mutex:
            for task in self.tasks:
                if task.task_data == task_data:
                    return task.task_status

            return None

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


class OperationWorkerThread:
    def __init__(self, thread_id, queue: TaskQueue):
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
                logging.info('THREAD {}: Attempting to terminate thread'.format(self.thread.name))
                self.process.terminate()
                self.process.join()
                logging.info('THREAD {}: Thread terminated'.format(self.thread.name))
                return True

            return False

    def run(self):
        logging.info('THREAD {}: Thread started'.format(self.thread.name))
        while True:
            logging.debug('THREAD {}: Waiting for new Task'.format(self.thread.name))
            try:
                with self.mutex:
                    self._current_task = self.queue.next_unprocessed(self.sleep_interval)
                    logging.info('THREAD {}: Running new task of type {}'.format(self.thread.name, type(self._current_task.task_data)))

                    queue = Queue()
                    self.process = Process(target=OperationWorkerThread._run_task, args=(self.thread.name, self._current_task, queue, ))
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
                logger.error("THREAD {}: Error in thread: {}".format(self.thread.name, e))
                if settings.DEBUG:  # only raise in debug mode to show stack-trace, as production server do not stop!
                    raise e
            else:  # Successfully finished!
                logging.debug('THREAD {}: Task finished successfully'.format(self.thread.name))
                self.queue.task_finished(self._current_task, result)
            finally:  # Always clear current data
                with self.mutex:
                    self.process = None
                    self._current_task = None

    @staticmethod
    def _run_task(name: str, task: Task, queue: Queue):
        try:
            start = time.time()
            task_data = task.task_data
            result = None
            if isinstance(task_data, TaskDataStaffLineDetection):
                data: TaskDataStaffLineDetection = task_data
                from omr.stafflines.detection.staffline_detector import create_staff_line_detector, StaffLineDetectorType, StaffLineDetector
                staff_line_detector: StaffLineDetector = create_staff_line_detector(StaffLineDetectorType.BASIC, data.page)
                result = staff_line_detector.detect(
                    data.page.file('binary_deskewed').local_path(),
                    data.page.file('gray_deskewed').local_path(),
                )
            else:
                logger.exception('THREAD {}: Unknown operation data {} of task {}'.format(name, task_data, task))

            logger.info("THREAD {}: Task finished. It ran for {}s".format(name, time.time() - start))
            queue.put(result)
        except Exception as e:
            logger.error("THREAD {}: Error in thread: {}".format(name, e))
            queue.put(e)


class OperationWorker:
    def __init__(self, num_threads=1):
        self.queue = TaskQueue()
        self.threads = [OperationWorkerThread(i, self.queue) for i in range(num_threads)]

    def stop(self, task_data):
        task = self.queue.remove(task_data)
        if task is not None:
            for i, t in enumerate(self.threads):
                if t.cancel_if_current_task(task):
                    break

    def put(self, task_data) -> bool:
        return self.queue.put(task_data)

    def pop_result(self, task_data) -> Any:
        return self.queue.pop_result(task_data)

    def status(self, task_data) -> Optional[TaskStatus]:
        return self.queue.status(task_data)


operation_worker = OperationWorker()


if __name__ == '__main__':
    print('done')
