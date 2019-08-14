from .taskqueue import TaskNotFinishedException
from .taskcommunicator import TaskCommunicationData
from .task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from multiprocessing import Queue, Process
import time
from omr.dataset.datafiles import EmptyDataSetException
import logging
from .taskresources import TaskResource
logger = logging.getLogger(__name__)


class TaskWorkerThread:
    def __init__(self, resource: TaskResource, task: Task, com_queue: Queue):
        self.resource = resource
        self.task = task
        self.com_queue = com_queue
        self.process = Process(target=TaskWorkerThread._run_task,
                               args=(self.task.task_id, self.task, self.com_queue,
                                     self.resource.gpu_id))
        self.process.daemon = False     # must be stopped explicitly
        self.process.start()

    def finished(self):
        return not self.process or not self.process.is_alive()

    def cancel(self) -> bool:
        if self.task is None:
            return False

        if self.process:
            logger.info('THREAD {}: Attempting to terminate thread'.format(self.process.name))
            self.process.terminate()
            self.process.join()
            logger.info('THREAD {}: Thread terminated'.format(self.process.name))
            return True

        return False

    @staticmethod
    def _run_task(name: str, task: Task, com_queue: Queue, gpu_id: int):
        logger.info('THREAD: Running new task {} of type {}'.format(task.task_id, type(task.task_runner)))
        
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
            if result is None:
                # process canceled
                raise TaskNotFinishedException()

            if isinstance(result, Exception):
                raise result
        except (BrokenPipeError, TaskNotFinishedException, EmptyDataSetException) as e:
            logger.info("THREAD {}: Task canceled".format(name))
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.ERROR)))
        except Exception as e:
            logger.exception("THREAD {}: Error in thread: {}".format(name, e))
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.ERROR), e))
        else:  # Successfully finished!
            logger.debug('THREAD {}: Task finished successfully'.format(name))
            com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.FINISHED), result))

        logger.debug("THREAD {}: Task exit.".format(name))
