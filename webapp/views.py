from django.shortcuts import render
from django.conf import settings
from django.utils.translation import get_language


def index(request, path=''):
    language = get_language()
    postfix = '-' + language if language != 'en' and language in [code for code, _ in settings.LANGUAGES] else ''
    return render(request, 'index.html', {'ommr4all_client': 'ommr4all-client' + postfix})
