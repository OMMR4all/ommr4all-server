from enum import IntEnum


class TaskWorkerGroup(IntEnum):
    LONG_TASKS_CPU = 0
    LONG_TASKS_GPU = 1
    NORMAL_TASKS_CPU = 2
    SHORT_TASKS_CPU = 3


