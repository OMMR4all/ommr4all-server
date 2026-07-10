import logging
from urllib.parse import parse_qs

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.layers import get_channel_layer

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.database_permissions import DatabaseBookPermissionFlag

logger = logging.getLogger(__name__)


def book_documents_group(book_id: str) -> str:
    return 'book_documents_{}'.format(book_id)


def notify_book_documents_changed(book_id: str):
    """Broadcast to all clients watching the chant list of the book.

    No-op when no channel layer is configured; never raises, live updates are best effort.
    """
    layer = get_channel_layer()
    if layer is None:
        return
    try:
        async_to_sync(layer.group_send)(book_documents_group(book_id), {'type': 'documents.changed'})
    except Exception:
        logger.exception('Failed to broadcast documents change of book {}'.format(book_id))


def update_book_documents_and_notify(book: DatabaseBook):
    """Refresh the (incrementally cached) documents of the book and broadcast a change
    notification if the chant structure actually differs.

    Best effort: a failure here must never fail the request that saved the page.
    """
    try:
        before = DatabaseBookDocuments.load(book).database_documents
        before_json = before.to_json() if before else None
        d = DatabaseBookDocuments.update_book_documents_cached(book)
        after_json = d.database_documents.to_json() if d.database_documents else None
        if before_json != after_json:
            notify_book_documents_changed(book.book)
    except Exception:
        logger.exception('Failed to update the documents of book {}'.format(book.book))


class TokenAuthMiddleware:
    """Authenticates websocket connections via a JWT passed as ?token= query parameter.

    Browsers cannot set an Authorization header on websocket connects, so the client
    appends its access token to the URL. Falls back to the (session) user already in
    the scope when no token is given.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        query = parse_qs(scope.get('query_string', b'').decode())
        token = (query.get('token') or [None])[0]
        if token:
            scope = dict(scope)
            scope['user'] = await _get_user_from_token(token)
        return await self.inner(scope, receive, send)


@database_sync_to_async
def _get_user_from_token(token: str):
    from django.contrib.auth.models import AnonymousUser
    from rest_framework_simplejwt.authentication import JWTAuthentication
    try:
        auth = JWTAuthentication()
        return auth.get_user(auth.get_validated_token(token))
    except Exception:
        return AnonymousUser()


class BookDocumentsConsumer(AsyncJsonWebsocketConsumer):
    """Notifies clients watching the chant (document) list of a book about changes.

    Messages are lightweight events; the client re-fetches the documents endpoint,
    which is cheap thanks to the per-page fragment cache.
    """

    group_name = None

    async def connect(self):
        book_id = self.scope['url_route']['kwargs']['book']
        if not await self._can_read(book_id, self.scope.get('user')):
            await self.close(code=4403)
            return
        self.group_name = book_documents_group(book_id)
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        if self.group_name:
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def documents_changed(self, event):
        await self.send_json({'event': 'documents_changed'})

    @database_sync_to_async
    def _can_read(self, book_id, user) -> bool:
        try:
            book = DatabaseBook(book_id)
            return book.exists() and user is not None \
                and book.resolve_user_permissions(user).has(DatabaseBookPermissionFlag.READ)
        except Exception:
            return False
