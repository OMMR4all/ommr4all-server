"""Entry point of the spawned task worker processes.

TaskWorkerThread starts one long-lived worker process per task resource using
the multiprocessing 'spawn' start method: the worker is a fresh interpreter,
so it can always initialize CUDA regardless of what the Django parent process
has imported (with the fork start method any torch/CUDA initialization in the
parent broke the child with "Cannot re-initialize CUDA in forked subprocess").

django.setup() must run at import time: the task payload contains Django model
instances (Task.creator) and may only be unpickled once the app registry is
populated.
"""
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
import django
django.setup()  # no-op in the already configured parent, required in the child

import logging
import pickle
import queue
import time
from typing import Optional

from .task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from .taskcommunicator import TaskCommunicationData
from .taskqueue import TaskNotFinishedException

logger = logging.getLogger(__name__)


def main(work_queue, com_queue, gpu_id: int, idle_timeout: Optional[float]):
    # must be set before anything imports torch: the worker only ever sees its
    # assigned GPU (or none for CPU resources)
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if gpu_id < 0 else str(gpu_id)
    logger.info('WORKER gpu={}: started (idle timeout: {}s)'.format(gpu_id, idle_timeout))

    while True:
        try:
            payload = work_queue.get(timeout=idle_timeout)
        except queue.Empty:
            try:
                # close the race between the idle timeout and a concurrent submit
                payload = work_queue.get_nowait()
            except queue.Empty:
                logger.info('WORKER gpu={}: idle, shutting down'.format(gpu_id))
                return

        _run_task(pickle.loads(payload), com_queue)


def _run_task(task: Task, com_queue):
    from omr.dataset.datafiles import EmptyDataSetException
    name = task.task_id
    logger.info('WORKER {}: Running new task of type {}'.format(name, type(task.task_runner)))
    try:
        start = time.time()
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.INITIALIZING)))
        result = task.task_runner.run(task, com_queue)
        logger.info('WORKER {}: Task finished. It ran for {}s'.format(name, time.time() - start))

        if result is None:
            # process canceled
            raise TaskNotFinishedException()

        if isinstance(result, Exception):
            raise result
    except (BrokenPipeError, TaskNotFinishedException, EmptyDataSetException) as e:
        logger.info('WORKER {}: Task canceled'.format(name))
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.ERROR), e))
    except Exception as e:
        logger.exception('WORKER {}: Error in worker: {}'.format(name, e))
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.ERROR), Exception('Internal error')))
    else:  # successfully finished
        logger.debug('WORKER {}: Task finished successfully'.format(name))
        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.FINISHED), result))

    logger.debug('WORKER {}: Task exit.'.format(name))
