from dataclasses import dataclass
from .taskworkergroup import TaskWorkerGroup
from typing import List
from multiprocessing import Value


class TaskResource:
    def __init__(self, group: TaskWorkerGroup, gpu_id: int = -1):
        self.group = group
        self.gpu_id = gpu_id
        self._used = Value('b', False)

    @property
    def used(self):
        return self._used.value
    
    @used.setter
    def used(self, v: bool):
        self._used.value = v


ResourcesList = List[TaskResource]


class Resources:
    def __init__(self, resources: ResourcesList = None):
        self.resources = resources if resources else []
        
    def free(self) -> ResourcesList:
        return [r for r in self.resources if not r.used]
    
    def used(self) -> ResourcesList:
        return [r for r in self.resources if r.used]
    
    def n_free(self) -> int:
        return len(self.free())
    
    def n_used(self) -> int:
        return len(self.used())

    def n_total(self) -> int:
        return len(self.resources)


def default_resources() -> Resources:
    import ommr4all.settings as settings
    return Resources([
               TaskResource(TaskWorkerGroup.LONG_TASKS_GPU, i) for i in settings.GPU_SETTINGS.available_gpus
           ] + [
               TaskResource(g) for g in ([TaskWorkerGroup.LONG_TASKS_CPU] * 2 +
                                         [TaskWorkerGroup.NORMAL_TASKS_CPU] * 2 +
                                         [TaskWorkerGroup.SHORT_TASKS_CPU] * 4)
           ])

