from typing import NamedTuple
from .task import Task, TaskStatus, TaskNotFoundException
from .taskqueue import TaskQueue
from .taskresources import Resources
import threading
import logging
import time

logger = logging.getLogger(__name__)


class TaskWatcher:
    def __init__(self, resources: Resources, task_queue: TaskQueue, interval_s: float):
        self.task_queue:TaskQueue = task_queue
        self.resources: Resources = resources
        self.interval = interval_s
        self.thread = threading.Thread(target=self.run, args=(), name='task_watcher')
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.thread.start()

    def run(self):
        logger.info("THREAD task_watcher: Started")
        while True:
            time.sleep(self.interval)
            try:
                status = self.task_queue.status()
                if status.n_total:
                    logger.info(
                        "State:\n" +
                        " - queue: {}\n".format(status) +
                        " - resources-free/used/total: {}/{}/{}\n".format(self.resources.n_free(), self.resources.n_used(), self.resources.n_total()) +
                        " - resources: {}".format([(r.group, r.used) for r in self.resources.resources]))
            except EOFError:
                pass
            except Exception as e:
                logger.exception(e)
                pass

