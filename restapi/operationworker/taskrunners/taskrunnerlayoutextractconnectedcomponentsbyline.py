from .taskrunner import TaskRunner, Queue, TaskWorkerGroup, Tuple
from database.database_page import DatabasePage
from ..taskcommunicator import TaskCommunicationData
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from database.file_formats.pcgts import Coords, PcGts


class TaskRunnerLayoutExtractConnectedComponentsByLine(TaskRunner):
    def __init__(self,
                 page: DatabasePage,
                 initial_line: Coords,
                 ):
        super().__init__({TaskWorkerGroup.SHORT_TASKS_CPU})
        self.page = page
        self.initial_line = initial_line

    def identifier(self) -> Tuple:
        return self.page, self.initial_line

    def run(self, task: Task, com_queue: Queue) -> dict:
        from omr.layout.correction_tools.connected_component_selector import extract_components
        import pickle
        staff_lines = []
        for mr in PcGts.from_file(self.page.file('pcgts')).page.music_regions:
            for ml in mr.staffs:
                staff_lines += ml.staff_lines

        with open(self.page.file('connected_components_deskewed', create_if_not_existing=True).local_path(), 'rb') as pkl:
            polys = extract_components(pickle.load(pkl), self.initial_line, staff_lines)

        return {'polys': [p.to_json() for p in polys]}
