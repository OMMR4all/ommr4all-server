from .taskqueue import TaskQueue, TaskNotFinishedException
from .taskcommunicator import TaskCommunicationData
from .task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from multiprocessing import Queue, Lock, Process
import threading
from queue import Empty as QueueEmptyException
import time
from typing import Set, TYPE_CHECKING
from .taskworkergroup import TaskWorkerGroup
from django.conf import settings
from omr.dataset.datafiles import EmptyDataSetException
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .operationworker import TaskResource


class TaskWorkerThread:
    gpu_available_ids = settings.GPU_SETTINGS.available_gpus
    gpu_id = 0

    def __init__(self, resource: 'TaskResource', task: Task, com_queue: Queue):
        self.resource = resource
        self.task = task
        self.com_queue = com_queue
        self.thread = threading.Thread(target=self.run, args=(), name="Thread for {}".format(task.task_id))
        self.thread.daemon = True  # daemon thread to stop automatically on shutdown
        self.process = None

        self.thread.start()

    def finished(self):
        return not self.thread.is_alive()

    def cancel(self) -> bool:
        if self.task is None:
            return False

        if self.process:
            logger.info('THREAD {}: Attempting to terminate thread'.format(self.thread.name))
            self.process.terminate()
            self.process.join()
            logger.info('THREAD {}: Thread terminated'.format(self.thread.name))
            return True

        return False

    def run(self):
        logger.info('THREAD {}: Running new task of type {}'.format(self.thread.name, type(self.task.task_runner)))
        try:

            queue = Queue()
            self.process = Process(target=TaskWorkerThread._run_task,
                                   args=(self.thread.name, self.task, queue, self.com_queue,
                                         self.resource.gpu_id))
            self.process.daemon = False     # must be stopped explicitly
            self.process.start()

            result = None

            # Try to fetch result
            while self.process.is_alive():
                try:
                    result = queue.get(timeout=1)   # get data every second
                except QueueEmptyException:
                    # no data written yet
                    continue

                break

            self.process.join()                     # wait for the task to finish

            if result is None:
                # process canceled
                raise TaskNotFinishedException()

            if isinstance(result, Exception):
                logger.info("THREAD {}: Exception during Task-Execution: {}".format(self.thread.name, result))
                raise result
        except (BrokenPipeError, TaskNotFinishedException, EmptyDataSetException) as e:
            logger.info("THREAD {}: Task canceled".format(self.thread.name))
            self.com_queue.put(TaskCommunicationData(self.task, TaskStatus(TaskStatusCodes.ERROR)))
        except Exception as e:
            logger.exception("THREAD {}: Error in thread: {}".format(self.thread.name, e))
            self.com_queue.put(TaskCommunicationData(self.task, TaskStatus(TaskStatusCodes.ERROR), e))
        else:  # Successfully finished!
            logger.debug('THREAD {}: Task finished successfully'.format(self.thread.name))
            self.com_queue.put(TaskCommunicationData(self.task, TaskStatus(TaskStatusCodes.FINISHED), result))

    @staticmethod
    def _run_task(name: str, task: Task, queue: Queue, com_queue: Queue, gpu_id: int):
        import os
        if gpu_id < 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            start = time.time()
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.INITIALIZING)))
            result = task.task_runner.run(task, com_queue)
            logger.info("THREAD {}: Task finished. It ran for {}s".format(name, time.time() - start))
            queue.put(result)
        except Exception as e:
            logger.exception("THREAD {}: Error in thread: {}".format(name, e))
            queue.put(e)

        logger.debug("THREAD {}: Task exit.".format(name))
