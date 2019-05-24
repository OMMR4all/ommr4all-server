from typing import NamedTuple
from .operation_worker import NamedTuple
from database.database_page import DatabasePage
from database.file_formats.pcgts import Coords, PcGts


class TaskDataExtractLayoutConnectedComponentByLine(NamedTuple):
    page: DatabasePage
    initial_line: Coords


class TaskExtractLayoutConnectedComponentByLineWorker:
    def run(self, data: TaskDataExtractLayoutConnectedComponentByLine) -> dict:
        from omr.layout.correction_tools.connected_component_selector import extract_components
        import pickle
        staff_lines = []
        for mr in PcGts.from_file(data.page.file('pcgts')).page.music_regions:
            for ml in mr.staffs:
                staff_lines += ml.staff_lines

        with open(data.page.file('connected_components_deskewed', create_if_not_existing=True).local_path(), 'rb') as pkl:
            polys = extract_components(pickle.load(pkl), data.initial_line, staff_lines)

        return {'polys': [p.to_json() for p in polys]}
