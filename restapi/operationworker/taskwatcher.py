"""Supervisor of the task scheduler.

The scheduler (TaskCreator) and the status pump (TaskCommunicator) are single daemon
threads inside the one mod_wsgi process, and both used to be able to die or wedge
without anyone noticing: tasks then queue up forever while the resource flags keep
reporting free slots, and only restarting the server helps. This watcher is what makes
that state recoverable in place -- it checks their heartbeats and restarts them.
"""
from .taskqueue import TaskQueue
from .taskresources import Resources
from typing import TYPE_CHECKING
import threading
import logging
import time

if TYPE_CHECKING:
    from .operationworker import OperationWorker

logger = logging.getLogger(__name__)


class TaskWatcher:
    def __init__(self, resources: Resources, task_queue: TaskQueue, interval_s: float,
                 worker: 'OperationWorker' = None, max_heartbeat_age: float = None):
        from ommr4all.settings import TASK_SCHEDULER_STALL_TIMEOUT
        self.task_queue: TaskQueue = task_queue
        self.resources: Resources = resources
        self.interval = interval_s
        self.worker = worker
        self.max_heartbeat_age = max_heartbeat_age if max_heartbeat_age is not None \
            else TASK_SCHEDULER_STALL_TIMEOUT
        self._stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(), name='task_watcher')
        self.thread.daemon = True  # daemon thread to stop automatically on shutdown
        self.thread.start()

    def stop(self):
        self._stopped.set()

    def check_once(self) -> dict:
        """One supervision pass. Separate from run() so it can be driven from tests.

        Only checks components that were actually created: the scheduler is started
        lazily on the first task, and importing Django (manage.py, migrations) must not
        bring one up.
        """
        result = {'creator_restarted': False, 'communicator_restarted': False}
        if self.worker is None:
            return result

        communicator = self.worker._task_communicator
        if communicator is not None and communicator.ensure_alive(self.max_heartbeat_age):
            result['communicator_restarted'] = True

        creator = self.worker._task_creator
        if creator is not None and creator.ensure_alive(self.max_heartbeat_age):
            result['creator_restarted'] = True

        if result['creator_restarted'] or result['communicator_restarted']:
            self._log_state(level=logging.ERROR)

        return result

    def _log_state(self, level=logging.INFO):
        status = self.task_queue.status()
        logger.log(level,
                   "State:\n" +
                   " - queue: {}\n".format(status) +
                   " - resources-free/used/quarantined/total: {}/{}/{}/{}\n".format(
                       self.resources.n_free(), self.resources.n_used(),
                       self.resources.n_quarantined(), self.resources.n_total()) +
                   " - resources: {}".format([(r.group, r.used, r.quarantined)
                                              for r in self.resources.resources]))

    def run(self):
        logger.info("THREAD task_watcher: Started")
        while not self._stopped.wait(self.interval):
            try:
                self.check_once()
                if self.task_queue.status().n_total:
                    self._log_state()
            except EOFError:
                pass
            except Exception as e:
                logger.exception(e)
                pass
