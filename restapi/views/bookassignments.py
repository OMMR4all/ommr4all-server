import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from django.contrib.auth.models import User
from django.db import DatabaseError
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from database import DatabaseBook
from database.database_book_assignments import BookAssignments, DatabaseBookAssignments, MAX_NOTE_LENGTH, \
    PageAssignment, sort_pages_in_book_order
from database.database_permissions import DatabaseBookPermissionFlag
from database.db_errors import log_db_failure
from restapi.models.auth import RestAPIUser
from restapi.models.error import APIError, ErrorCodes
from restapi.views.bookaccess import etag_response, require_permissions

logger = logging.getLogger(__name__)

_LOCK_LABELS = ['StaffLines', 'Layout', 'Symbols', 'Text']


def _progress_of(pages: List[str], existing_pages: set, rows: Dict[str, 'object']) -> dict:
    """Assignment progress derived from the page index (never from page_progress.json).

    Existence comes from the page folders on disk, progress from the stored index rows: a
    page that exists but has no row yet counts as untouched, not as missing.

    The buckets must match the client's PageEditingProgress: finished = all four locks,
    inProgress = at least one but not all, untouched = none."""
    progress = {'total': len(pages), 'existing': 0, 'missing': 0,
                'untouched': 0, 'inProgress': 0, 'finished': 0, 'verified': 0,
                'locks': {label: 0 for label in _LOCK_LABELS}}
    for page in pages:
        if page not in existing_pages:
            progress['missing'] += 1
            continue
        progress['existing'] += 1
        row = rows.get(page)
        if row is None:
            progress['untouched'] += 1
            continue
        locks = row.progress_locks or {}
        set_locks = [label for label in _LOCK_LABELS if locks.get(label, False)]
        for label in set_locks:
            progress['locks'][label] += 1
        if len(set_locks) == len(_LOCK_LABELS):
            progress['finished'] += 1
        elif len(set_locks) > 0:
            progress['inProgress'] += 1
        else:
            progress['untouched'] += 1
        if row.verified:
            progress['verified'] += 1
    return progress


def _assignment_to_json(book: DatabaseBook, assignment: PageAssignment, users: Dict[str, User],
                        existing_pages: set, rows: Dict[str, 'object']) -> dict:
    user = users.get(assignment.username)
    return {
        'id': assignment.id,
        'user': (RestAPIUser.from_user(user) if user else RestAPIUser(assignment.username)).to_dict(),
        'userExists': user is not None,
        # assigning a user who cannot read the book yet is allowed (access is often granted
        # afterwards), but the overview shows a warning for it
        'userHasAccess': user is not None
        and book.resolve_user_permissions(user).has(DatabaseBookPermissionFlag.READ),
        'pages': assignment.pages,
        'note': assignment.note,
        'created': assignment.created.isoformat() if assignment.created else None,
        'createdBy': assignment.createdBy,
        'updated': assignment.updated.isoformat() if assignment.updated else None,
        'updatedBy': assignment.updatedBy,
        'progress': _progress_of(assignment.pages, existing_pages, rows),
    }


def _currently_editing(book: DatabaseBook) -> List[dict]:
    """Live edit locks of the book -- who has a page open right now.

    Not to be confused with DatabaseBook.pages_with_lock, which filters *progress* locks."""
    from database.database_page import edit_lock_is_stale
    from database.models.book_index import PageEditLock
    try:
        locks = PageEditLock.objects.filter(page__book__name=book.book).select_related('user', 'page')
        # stale locks are expired on the page path, this read path only skips them
        return [{'page': lock.page.name,
                 'user': RestAPIUser.from_user(lock.user).to_dict(),
                 'since': lock.acquired_at.isoformat()}
                for lock in locks if not edit_lock_is_stale(lock)]
    except DatabaseError as e:
        log_db_failure(logger, 'Resolving the edit locks of {}'.format(book.book), e)
        return []


def _page_rows(book: DatabaseBook, sync: bool) -> Dict[str, 'object']:
    """Index rows of the book's pages, keyed by page label.

    `sync=False` (the default for every consumer that only highlights pages) reads the
    stored rows: no stat loop and no writes, which matters because the editor sidebar and
    the content view hit this endpoint once per open browser tab. The overview asks for
    `?sync=1`, where up-to-date progress is the whole point.

    A database failure degrades to 'no progress information' rather than failing the
    request -- the assignments themselves live in the storage folder, not in the index."""
    from database.book_index import stored_book_pages, sync_book_pages
    try:
        rows = sync_book_pages(book) if sync else stored_book_pages(book)
        return {row.name: row for row in rows}
    except DatabaseError as e:
        log_db_failure(logger, 'Reading the page index of {}'.format(book.book), e)
        return {}


def _users(usernames) -> Dict[str, User]:
    try:
        return {u.username: u for u in User.objects.filter(username__in=set(usernames))}
    except DatabaseError as e:
        log_db_failure(logger, 'Resolving the assignees {}'.format(sorted(set(usernames))), e)
        return {}


def _payload(request, book: DatabaseBook, stored: BookAssignments, sync: bool = False) -> dict:
    # existence and order come from the page folders (one scandir), progress from the index
    page_names = book.page_names_on_disk()
    existing_pages = set(page_names)
    rows = _page_rows(book, sync)
    users = _users([a.username for a in stored.assignments])
    assignments = [_assignment_to_json(book, a, users, existing_pages, rows)
                   for a in stored.assignments]
    assigned_pages = {p for a in stored.assignments for p in a.pages if p in existing_pages}
    return {
        'assignments': assignments,
        'currentlyEditing': _currently_editing(book),
        # the ordered page labels power the jump button, the range compaction and the
        # dialog's page pickers, so no client needs an extra listPages call
        'pageOrder': page_names,
        'permissions': book.resolve_user_permissions(request.user).flags,
        'totalPages': len(page_names),
        'assignedPages': len(assigned_pages),
    }


def _single_response(request, book: DatabaseBook, assignment: PageAssignment,
                     http_status=status.HTTP_200_OK) -> Response:
    existing_pages = set(book.page_names_on_disk())
    rows = _page_rows(book, sync=False)
    users = _users([assignment.username])
    return Response(_assignment_to_json(book, assignment, users, existing_pages, rows),
                    status=http_status)


def _parse_body(request, book: DatabaseBook):
    """(username, pages, note) or an APIError response."""
    body = json.loads(request.body)
    username = (body.get('username') or '').strip()
    if not User.objects.filter(username=username).exists():
        return None, APIError(status.HTTP_406_NOT_ACCEPTABLE,
                              'Unknown user {}'.format(username),
                              'The user {} does not exist.'.format(username),
                              ErrorCodes.BOOK_ASSIGNMENT_UNKNOWN_USER,
                              ).response()

    pages = list(dict.fromkeys(body.get('pages', [])))
    existing = set(book.page_names_on_disk())
    unknown = [p for p in pages if p not in existing]
    if unknown:
        return None, APIError(status.HTTP_406_NOT_ACCEPTABLE,
                              'Unknown pages {} in book {}'.format(unknown, book.book),
                              'The page(s) {} do(es) not exist.'.format(', '.join(unknown)),
                              ErrorCodes.BOOK_ASSIGNMENT_UNKNOWN_PAGE,
                              ).response()

    return (username, sort_pages_in_book_order(book, pages),
            (body.get('note') or '')[:MAX_NOTE_LENGTH]), None


class BookAssignmentsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        db_book = DatabaseBook(book)
        sync = request.query_params.get('sync', '0') not in ('0', '', 'false')
        return etag_response(request, _payload(request, db_book,
                                               DatabaseBookAssignments.load(db_book), sync))

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def put(self, request, book):
        db_book = DatabaseBook(book)
        parsed, error = _parse_body(request, db_book)
        if error:
            return error
        username, pages, note = parsed

        assignment = PageAssignment(id=uuid.uuid4().hex, username=username, pages=pages, note=note,
                                    created=datetime.now(), createdBy=request.user.username)

        def apply(assignments: BookAssignments):
            assignments.assignments.append(assignment)

        DatabaseBookAssignments.mutate(db_book, apply)
        return _single_response(request, db_book, assignment, status.HTTP_201_CREATED)


class BookAssignmentView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def post(self, request, book, id):
        db_book = DatabaseBook(book)
        parsed, error = _parse_body(request, db_book)
        if error:
            return error
        username, pages, note = parsed

        def apply(assignments: BookAssignments) -> Optional[PageAssignment]:
            assignment = assignments.by_id(id)
            if assignment is None:
                return None
            assignment.username = username
            assignment.pages = pages
            assignment.note = note
            assignment.updated = datetime.now()
            assignment.updatedBy = request.user.username
            return assignment

        assignment = DatabaseBookAssignments.mutate(db_book, apply)
        if assignment is None:
            return self._not_found(id, book)
        return _single_response(request, db_book, assignment)

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def delete(self, request, book, id):
        db_book = DatabaseBook(book)

        def apply(assignments: BookAssignments) -> bool:
            assignment = assignments.by_id(id)
            if assignment is None:
                return False
            assignments.assignments.remove(assignment)
            return True

        if not DatabaseBookAssignments.mutate(db_book, apply):
            return self._not_found(id, book)
        return Response(status=status.HTTP_200_OK)

    @staticmethod
    def _not_found(id, book):
        # 406, not 404: the root urlconf wraps the app in i18n_patterns, so LocaleMiddleware
        # turns every 404 into a 302 language redirect that the client would follow
        return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                        'Assignment {} not found in book {}'.format(id, book),
                        'The assignment does not exist (anymore).',
                        ErrorCodes.BOOK_ASSIGNMENT_NOT_FOUND,
                        ).response()
