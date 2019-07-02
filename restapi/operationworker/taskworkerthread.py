from .taskqueue import TaskQueue, TaskNotFinishedException
from .taskcommunicator import TaskCommunicationData
from .task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from multiprocessing import Queue, Lock, Process
import threading
from queue import Empty as QueueEmptyException
import time
from typing import Set
from .taskworkergroup import TaskWorkerGroup
from django.conf import settings
from omr.dataset.datafiles import EmptyDataSetException
import logging
logger = logging.getLogger(__name__)


class TaskWorkerThread:
    gpu_available_ids = settings.GPU_SETTINGS.available_gpus
    gpu_id = 0

    def __init__(self, thread_id, queue: TaskQueue, com_queue: Queue, groups: Set[TaskWorkerGroup]):
        self.groups = groups
        if TaskWorkerGroup.LONG_TASKS_GPU in groups and len(TaskWorkerThread.gpu_available_ids) > TaskWorkerThread.gpu_id:
            # select gpu if available
            self.gpu_id = TaskWorkerThread.gpu_available_ids[TaskWorkerThread.gpu_id]
            TaskWorkerThread.gpu_id += 1
        else:
            self.gpu_id = -1

        self.com_queue = com_queue
        self.thread = threading.Thread(target=self.run, args=(), name=thread_id)
        self.thread.daemon = True  # daemon thread to stop automatically on shutdown
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
                    self._current_task = self.queue.next_unprocessed(self.groups, self.sleep_interval)
                    logger.info('THREAD {}: Running new task of type {}'.format(self.thread.name, type(self._current_task.task_runner)))

                    queue = Queue()
                    self.process = Process(target=TaskWorkerThread._run_task,
                                           args=(self.thread.name, self._current_task, queue, self.com_queue,
                                                 self.gpu_id))
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
