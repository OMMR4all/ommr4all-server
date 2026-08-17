import atexit
import os
from typing import Optional, TYPE_CHECKING
from .taskqueue import TaskQueue, TaskStatus
from .taskcommunicator import TaskCommunicator
from uuid import uuid4, UUID
from .taskresources import Resources, default_resources
from .taskrunners.taskrunner import TaskRunner
import logging
from .taskcreator import TaskCreator
from .taskwatcher import TaskWatcher
from ommr4all.settings import TASK_OPERATION_WATCHER_SETTINGS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from django.contrib.auth.models import User


class TaskIDGenerator:
    def gen(self):
        return str(UUID(bytes=os.urandom(16), version=4))


class OperationWorker:
    def __init__(self, resources: Resources = None, watcher_interval=TASK_OPERATION_WATCHER_SETTINGS.interval):
        self.queue = TaskQueue()
        self.resources = resources if resources else default_resources()
        self._task_communicator: Optional[TaskCommunicator] = None
        self._task_creator: Optional[TaskCreator] = None
        self.id_generator = TaskIDGenerator()
        self.task_watcher: Optional[TaskWatcher] = None
        # <= 0 exists so tests can build an unsupervised worker and drive check_once()
        # by hand; in production the watcher is what recovers a wedged scheduler
        if watcher_interval > 0:
            self.task_watcher = TaskWatcher(self.resources, self.queue, watcher_interval, worker=self)

    def task_communicator(self) -> TaskCommunicator:
        if not self._task_communicator:
            self._task_communicator = TaskCommunicator(self.queue)
        return self._task_communicator

    def task_creator(self) -> TaskCreator:
        if not self._task_creator:
            self._task_creator = TaskCreator(self.queue, self.task_communicator(), self.resources)
        return self._task_creator

    def id_by_task_runner(self, task_runner: TaskRunner):
        return self.queue.id_by_runner(task_runner)

    def stop(self, task_id: str):
        task = self.queue.remove(task_id)
        if task is not None:
            self.task_creator().stop(task)

    def put(self, task_runner: TaskRunner, creator: 'User') -> str:
        self.task_creator()  # require creation
        task_id = self.id_generator.gen()
        self.queue.put(task_id, task_runner, creator)
        return task_id

    def pop_result(self, task_id: str) -> dict:
        return self.queue.pop_result(task_id)

    def status(self, task_id) -> Optional[TaskStatus]:
        return self.queue.status_of_task(task_id)

    def health(self) -> dict:
        """Liveness of the scheduler machinery, for the admin view.

        The worker slots alone cannot answer 'can this server still start a task?' --
        they are bookkeeping the scheduler writes, so a dead scheduler keeps reporting
        free slots. A scheduler that was never needed yet counts as healthy.
        """
        from .taskworkerthread import unreaped_workers
        creator = self._task_creator
        communicator = self._task_communicator
        return {
            'scheduler': creator.state() if creator else {'alive': True, 'heartbeatAge': 0.0,
                                                          'iterations': 0, 'restarts': 0, 'nRunning': 0},
            'communicator': communicator.state() if communicator else {'alive': True, 'heartbeatAge': 0.0,
                                                                       'restarts': 0},
            'started': creator is not None,
            'healthy': (creator is None or creator.healthy()) and (communicator is None or communicator.healthy()),
            'nQuarantined': self.resources.n_quarantined(),
            'unreapedWorkers': unreaped_workers(),
        }

    def release_resource(self, index: int, fail_task: bool = True) -> bool:
        """Force worker slot ``index`` back into service (admin recovery)."""
        if index < 0 or index >= len(self.resources.resources):
            return False
        self.task_creator().release_resource(self.resources.resources[index], fail_task=fail_task)
        return True

    def shutdown(self) -> None:
        """Stop the supervision and scheduling threads.

        Called from atexit and from tests. Daemon threads are not enough on their own:
        one that is still logging while the interpreter finalizes takes the process down
        with 'could not acquire lock for <stdout>'.
        """
        if self.task_watcher is not None:
            self.task_watcher.stop()
        if self._task_creator is not None:
            self._task_creator.stop_loop()
        if self._task_communicator is not None:
            self._task_communicator.stop_loop()

    def repair(self) -> dict:
        """Restart dead/stalled scheduler threads and reclaim leaked slots."""
        if self._task_communicator is not None:
            self._task_communicator.ensure_alive()
        if self._task_creator is not None:
            self._task_creator.ensure_alive()
            self._task_creator.reconcile()
        return self.health()


operation_worker = OperationWorker()

# registered before taskworkerthread's own atexit handler runs (atexit is LIFO, and that
# module is imported later), so the loops are quiet by the time the workers are torn down
atexit.register(operation_worker.shutdown)
