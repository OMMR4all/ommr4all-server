"""Select image-diverse page assignments inside a book-level file lock."""
import logging
import os
import uuid
from datetime import datetime
from typing import Tuple

from database import DatabaseBook
from database.book_index import prefill_page_progress
from database.database_book_assignments import (DatabaseBookAssignments, PageAssignment,
                                                  sort_pages_in_book_order)
from database.file_formats.performance.pageprogress import Locks
from omr.discovery.config import FeatureConfig
from omr.discovery.features import build_feature_extractor
from omr.discovery.page_selection import (UnreadableOriginalImage, select_representative_pages,
                                          whole_image_page_embedding)
from .pageselection import PageSelection
from .taskrunner import TaskRunner, Queue, TaskWorkerGroup
from ..task import Task, TaskStatus, TaskStatusCodes, TaskProgressCodes
from ..taskcommunicator import TaskCommunicationData

logger = logging.getLogger(__name__)


class InsufficientSuggestedPages(Exception):
    def __init__(self, available: int):
        self.available = available
        super().__init__('Only {} eligible image-backed pages remain'.format(available))


def _finished(page) -> bool:
    progress = page.page_progress()
    return progress.verified or all(progress.locked.get(lock, False) for lock in Locks)


class TaskRunnerSuggestedAssignment(TaskRunner):
    operation_name = 'suggested_self_assignment'

    def __init__(self, book: DatabaseBook, username: str, count: int, resources=None,
                 allow_unavailable: bool = False, from_page: str = None, to_page: str = None):
        if count < 1:
            raise ValueError('suggested assignment count must be positive')
        if (from_page is None) != (to_page is None):
            raise ValueError('both page range endpoints are required')
        self.range_pages = None
        if from_page is not None:
            order = book.page_names_on_disk()
            if from_page not in order or to_page not in order:
                raise ValueError('page range endpoints must exist in the book')
            start, end = sorted((order.index(from_page), order.index(to_page)))
            self.range_pages = frozenset(order[start:end + 1])
        if resources is None:
            from restapi.operationworker.operationworker import operation_worker
            resources = operation_worker.resources
        groups = {resource.group for resource in resources.resources
                  if not resource.quarantined or allow_unavailable}
        if TaskWorkerGroup.LONG_TASKS_GPU in groups:
            group = TaskWorkerGroup.LONG_TASKS_GPU
        elif TaskWorkerGroup.LONG_TASKS_CPU in groups:
            group = TaskWorkerGroup.LONG_TASKS_CPU
        else:
            if not allow_unavailable:
                raise RuntimeError('No long CPU or GPU worker is configured')
            group = TaskWorkerGroup.LONG_TASKS_CPU
        super().__init__(None, PageSelection.from_book(book), [group])
        self.book = book
        self.username = username
        self.count = count

    def identifier(self) -> Tuple:
        return self.book.book, self.username

    def run(self, task: Task, com_queue: Queue) -> dict:
        result = self._run_for_users(task, com_queue, [task.creator.username])
        if 'assignment_ids' in result:
            result['assignment_id'] = result.pop('assignment_ids')[0]
        return result

    def _run_for_users(self, task: Task, com_queue: Queue, usernames: list[str]) -> dict:
        pages = prefill_page_progress(self.book)
        if self.range_pages is not None:
            pages = [page for page in pages if page.page in self.range_pages]
        # Fail before expensive inference if the assignment file cannot be parsed.
        DatabaseBookAssignments.load(self.book, strict=True)
        images = [page for page in pages
                  if os.path.isfile(page.file('color_original').local_path())]
        n_total = len(images)
        embeddings = {}
        excluded = [{'page': page.page, 'reason': 'missing_original_image'}
                    for page in pages if not os.path.isfile(
                        page.file('color_original').local_path())]
        extractor = build_feature_extractor(FeatureConfig(backend='dino', device='auto',
                                                          max_input_side=448))
        extractor.load()
        for index, page in enumerate(images, 1):
            try:
                embeddings[page.page] = whole_image_page_embedding(page, extractor)
            except FileNotFoundError:
                excluded.append({'page': page.page, 'reason': 'missing_original_image'})
            except UnreadableOriginalImage as exc:
                logger.warning('Unreadable original image %s: %s', page.page, exc)
                excluded.append({'page': page.page, 'reason': 'unreadable_original_image'})
            com_queue.put(TaskCommunicationData(task, TaskStatus(
                TaskStatusCodes.RUNNING, TaskProgressCodes.WORKING,
                progress=index / n_total, n_processed=index, n_total=n_total)))

        def apply(assignments):
            # Re-read progress under the same file lock as the assignment write. These are
            # fresh page objects, not the pre-inference progress cache.
            current = prefill_page_progress(self.book)
            if self.range_pages is not None:
                current = [page for page in current if page.page in self.range_pages]
            owned = {name for assignment in assignments.assignments for name in assignment.pages}
            finished = {page.page for page in current if _finished(page)}
            existing = {page.page for page in current}
            candidates = sorted((existing & embeddings.keys()) - owned - finished)
            total = self.count * len(usernames)
            if len(candidates) < total:
                raise InsufficientSuggestedPages(len(candidates))
            references = (owned | finished) & embeddings.keys()
            selected = select_representative_pages(embeddings, candidates, references, total)
            now = datetime.now()
            created = []
            for index, username in enumerate(usernames):
                assignment = PageAssignment(
                    id=uuid.uuid4().hex, username=username,
                    pages=sort_pages_in_book_order(
                        self.book, [item.page for item in selected[index::len(usernames)]]),
                    created=now, createdBy=task.creator.username)
                assignments.assignments.append(assignment)
                created.append(assignment.id)
            return created

        try:
            assignment_ids = DatabaseBookAssignments.mutate(self.book, apply)
        except InsufficientSuggestedPages as exc:
            return {'error': 'insufficient_pages', 'available': exc.available,
                    'excluded_pages': excluded}
        return {'assignment_ids': assignment_ids, 'excluded_pages': excluded}


class TaskRunnerSuggestedBatchAssignment(TaskRunnerSuggestedAssignment):
    operation_name = 'suggested_batch_assignment'

    def __init__(self, book: DatabaseBook, creator: str, usernames: list[str], count: int,
                 resources=None, allow_unavailable: bool = False,
                 from_page: str = None, to_page: str = None):
        super().__init__(book, creator, count, resources, allow_unavailable, from_page, to_page)
        self.usernames = list(usernames)

    def run(self, task: Task, com_queue: Queue) -> dict:
        return self._run_for_users(task, com_queue, self.usernames)
