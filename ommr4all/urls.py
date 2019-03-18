from django.conf import settings
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns

from django.conf.urls.static import static

import os

urlpatterns = \
    [
        path('admin/', admin.site.urls),
        path('api/', include('restapi.urls')),
    ] \
    + i18n_patterns(
        path('', include('webapp.urls'))
    ) \
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) \
    + static('assets', document_root=os.path.join(settings.STATIC_ROOT, 'ommr4all-client', 'assets')) \
