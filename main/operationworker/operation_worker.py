import threading
import time
from typing import NamedTuple, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, asdict

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
        self.mutex = threading.Lock()

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
    def __init__(self, queue: TaskQueue):
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()
        self.sleep_interval = 1.0  # seconds
        self.queue = queue

    def run(self):
        from omr.stafflines.detection.staffline_detector import create_staff_line_detector, StaffLineDetectorType, StaffLineDetector
        while True:
            task = self.queue.next_unprocessed(self.sleep_interval)
            logging.info('Running new task of type {}'.format(type(task.task_data)))
            try:
                start = time.time()
                task_data = task.task_data
                result = None
                if isinstance(task_data, TaskDataStaffLineDetection):
                    data: TaskDataStaffLineDetection = task_data
                    staff_line_detector: StaffLineDetector = create_staff_line_detector(StaffLineDetectorType.BASIC, data.page)
                    result = staff_line_detector.detect(
                        data.page.file('binary_deskewed').local_path(),
                        data.page.file('gray_deskewed').local_path(),
                    )
                else:
                    logger.exception("Unknown operation data {} of task {}".format(task_data, task))

                logger.info("Task finished. It ran for {}s".format(time.time() - start))
                self.queue.task_finished(task, result)
            except Exception as e:
                import traceback
                logger.error("Error in thread: {}".format(e))
                self.queue.task_error(task, e)
                # TODO: only reraise if debug to stop thread
                raise e


class OperationWorker:
    def __init__(self, num_threads=1):
        self.queue = TaskQueue()
        self.threads = [OperationWorkerThread(self.queue) for _ in range(num_threads)]

    def put(self, task_data) -> bool:
        return self.queue.put(task_data)

    def pop_result(self, task_data) -> Any:
        return self.queue.pop_result(task_data)

    def status(self, task_data) -> Optional[TaskStatus]:
        return self.queue.status(task_data)


operation_worker = OperationWorker()


if __name__ == '__main__':
    print('done')
