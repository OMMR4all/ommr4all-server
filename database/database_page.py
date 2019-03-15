from database.database_book import DatabaseBook, file_name_validator, InvalidFileNameException
import os
import shutil


class DatabasePage:
    def __init__(self, book: DatabaseBook, page: str, skip_validation=False):
        self.book = book
        self.page = page.strip("/")
        if not skip_validation and not file_name_validator.fullmatch(self.page):
            raise InvalidFileNameException(self.page)

    def __eq__(self, other):
        return isinstance(other, DatabasePage) and self.book == other.book and self.page == other.page

    def delete(self):
        if os.path.exists(self.local_path()):
            shutil.rmtree(self.local_path())

    def rename(self, new_name):
        if not file_name_validator.fullmatch(new_name):
            raise InvalidFileNameException(new_name)

        old_path = self.local_path()
        self.page = new_name
        new_path = self.local_path()

        shutil.move(old_path, new_path)

    def file(self, fileId, create_if_not_existing=False):
        from database.database_file import DatabaseFile
        return DatabaseFile(self, fileId, create_if_not_existing)

    def local_file_path(self, f):
        return os.path.join(self.local_path(), f)

    def local_path(self):
        return os.path.join(self.book.local_path('pages'), self.page)

    def remote_path(self):
        return os.path.join(self.book.remote_path(), self.page)

    def is_valid(self):
        if not os.path.exists(self.local_path()):
            return True

        if not os.path.isdir(self.local_path()):
            return False

        return True


class DatabasePageMeta:
    def __init__(self, d: dict = None):
        d = d if d else {}
        self.width = d.get('width', -1)
        self.height = d.get('height', -1)

    def json(self):
        return {
            'width': self.width,
            'height': self.height,
        }

    def dumpfn(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.json(), f)

    @staticmethod
    def loadfn(filename):
        import json
        with open(filename, 'r') as f:
            return DatabasePageMeta(json.load(f))

