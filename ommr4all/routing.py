from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application
from django.urls import path

# The Django ASGI app must be instantiated (django.setup) before importing consumers,
# which pull in models.
django_asgi_app = get_asgi_application()

from restapi.consumers import BookDocumentsConsumer, TokenAuthMiddleware  # noqa: E402

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            TokenAuthMiddleware(
                URLRouter([
                    path("ws/book/<str:book>/documents/", BookDocumentsConsumer.as_asgi()),
                ])
            )
        )
    ),
})
