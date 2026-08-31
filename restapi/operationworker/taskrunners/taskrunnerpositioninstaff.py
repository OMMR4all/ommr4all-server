from typing import Tuple

from database import DatabaseBook, DatabasePage
from database.file_formats.performance.pageprogress import Locks
from database.file_formats.pcgts import SymbolType
from .pageselection import PageSelection
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from ..taskcommunicator import TaskCommunicationData
import logging

logger = logging.getLogger(__name__)


class TaskRunnerPositionInStaff(TaskRunner):
    """Re-derives the position in staff of every symbol of a book from its coordinates.

    The stored position wins over the geometry everywhere else (see MusicSymbol.update_note_name),
    so changed pitch detection parameters only reach the existing pages through this task.
    Pages a user has locked or verified keep what they contain.
    """

    operation_name = 'reapply_position_in_staff'

    def __init__(self, book: DatabaseBook):
        super().__init__(None, PageSelection.from_book(book), [TaskWorkerGroup.SHORT_TASKS_CPU])
        self.book = book

    def identifier(self) -> Tuple:
        return self.book.book,

    @staticmethod
    def unprocessed(page: DatabasePage) -> bool:
        return True

    @staticmethod
    def _locked(page: DatabasePage) -> bool:
        progress = page.page_progress()
        return progress.verified or bool(progress.locked.get(Locks.SYMBOLS))

    def run(self, task: Task, com_queue: Queue) -> dict:
        # one index backed pass, so that the lock of every page is known without parsing
        # page_progress.json once per page
        from database.book_index import prefill_page_progress
        pages = prefill_page_progress(self.book)
        n_total = len(pages)

        def progress(n_processed: int):
            com_queue.put(TaskCommunicationData(task, TaskStatus(
                TaskStatusCodes.RUNNING,
                TaskProgressCodes.WORKING,
                progress=n_processed / n_total if n_total > 0 else 1,
                n_processed=n_processed,
                n_total=n_total,
            )))

        progress(0)

        n_updated, n_skipped, n_failed, n_symbols_changed = 0, 0, 0, 0
        for i, page in enumerate(pages):
            try:
                if self._locked(page):
                    n_skipped += 1
                    continue

                changed = 0
                pcgts = page.pcgts()
                for line in pcgts.page.all_music_lines():
                    for symbol in line.symbols:
                        pis = line.staff_lines.compute_position_in_staff(
                            symbol.coord, clef=symbol.symbol_type == SymbolType.CLEF)
                        if pis != symbol.position_in_staff:
                            symbol.position_in_staff = pis
                            changed += 1

                if changed > 0:
                    # the pitches are derived from the positions, so they are stale now
                    pcgts.page.update_note_names()
                    pcgts.to_file(page.file('pcgts').local_path())
                    page.mark_updated(task.creator)
                    n_updated += 1
                    n_symbols_changed += changed
            except Exception as e:
                # one broken page must not abort the whole book
                n_failed += 1
                logger.warning('Could not re-derive the positions in staff of page {}/{}'.format(
                    self.book.book, page.page))
                logger.exception(e)
            finally:
                progress(i + 1)

        com_queue.put(TaskCommunicationData(task, TaskStatus(TaskStatusCodes.RUNNING, TaskProgressCodes.FINALIZING)))
        return {
            'n_total': n_total,
            'n_updated': n_updated,
            'n_skipped': n_skipped,
            'n_failed': n_failed,
            'n_symbols_changed': n_symbols_changed,
        }
