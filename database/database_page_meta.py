from dataclasses import dataclass
from database.database_page import DatabasePage
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass()
class Preprocessing(DataClassJSONMixin):
    auto_line_distance: bool = True
    average_line_distance: int = -1
    deskewing_degrees: float = 0
    deskew: bool = True


@dataclass
class DatabasePageMeta(DataClassJSONMixin):
    preprocessing: Preprocessing

    @staticmethod
    def load(page: DatabasePage):
        path = page.file('meta').local_path()
        try:
            with open(path) as f:
                return DatabasePageMeta.from_json(f.read())
        except FileNotFoundError as e:
            return DatabasePageMeta(
                Preprocessing()
            )

    def save(self, page: DatabasePage):
        dump = self.to_json()
        with open(page.file('meta').local_path(), 'w') as f:
            f.write(dump)
