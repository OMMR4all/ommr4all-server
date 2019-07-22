from django.shortcuts import render
from django.conf import settings
from django.utils.translation import get_language
import os
import re
import logging
logger = logging.getLogger(__name__)

# load the currently required scripts of the webapp from the client's index.html
# do this dynamically since the hashes are included

script_search = re.compile(r"<script src=\"(\S*)\" (\S*)></script>")
styles_search = re.compile(r"<link rel=\"stylesheet\" href=\"(\S*)\">")


def extract_content(language_code):
    client_dir = os.path.join(settings.BASE_DIR, 'webapp', 'static', 'ommr4all-client')
    if language_code != 'en':
        client_dir += "-" + language_code

    scripts = []
    styles = []
    with open(os.path.join(client_dir, 'index.html'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("<link rel=\"stylesheet\""):
                for m in styles_search.finditer(line):
                    styles.append({'href': m.group(1)})
            elif line.startswith("<script src="):
                for m in script_search.finditer(line):
                    scripts.append({'src': m.group(1), 'module': m.group(2)})
    return scripts, styles


try:
    files = {lc: extract_content(lc) for lc, _ in settings.LANGUAGES}
except FileNotFoundError as e:
    logger.info("Could not find index.html files of the deployed webapp."
                " Possibly running tests and not deployment.")
    files = None


def index(request, path=''):
    if not files:
        raise Exception("index.html files not parsed properly!")

    language = get_language()
    postfix = '-' + language if language != 'en' and language in [code for code, _ in settings.LANGUAGES] else ''
    scripts, styles = files[language]

    return render(request, 'index.html', {
        'ommr4all_client': 'ommr4all-client' + postfix + '/',
        'lc': language,
        'scripts': scripts,
        'styles': styles,
    })
