from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage
from ..task import Task
from database.file_formats.pcgts import Coords, PcGts, PageScaleReference
from typing import List


class TaskRunnerLayoutExtractConnectedComponentsByLine(TaskRunner):
    def __init__(self,
                 pcgts: PcGts,
                 initial_line: Coords,
                 ):
        super().__init__({TaskWorkerGroup.SHORT_TASKS_CPU})
        self.pcgts = pcgts
        self.page = self.pcgts.page.location
        self.initial_line = initial_line

    def identifier(self) -> Tuple:
        return self.pcgts.page.location, self.initial_line

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return True

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.layout.correction_tools.connected_component_selector import extract_components
        import pickle
        staff_lines: List[Coords] = []
        pcgts = self.pcgts
        for mr in pcgts.page.music_blocks():
            for ml in mr.lines:
                staff_lines += [pcgts.page.page_to_image_scale(s.coords, PageScaleReference.NORMALIZED) for s in ml.staff_lines]

        with open(self.page.file('connected_components_norm', create_if_not_existing=True).local_path(), 'rb') as pkl:
            polys = extract_components(pickle.load(pkl), pcgts.page.page_to_image_scale(self.initial_line, PageScaleReference.NORMALIZED), staff_lines)
            polys = [pcgts.page.image_to_page_scale(c, PageScaleReference.NORMALIZED) for c in polys]

        return {'polys': [p.to_json() for p in polys]}
