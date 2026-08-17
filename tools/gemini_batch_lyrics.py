"""Transcribe the lyrics of a page range of a book with the Gemini Batch API.

The tool is split into commands so that a run survives a crash, a lost network
connection or simply a closed terminal: the batch job names and the page they
belong to are stored in a state file (``batch_state.json`` inside the output
directory) and every command reads/updates it. The output directory is self
contained and may be moved or copied to another machine, where the run can be
continued with the same API key (batch jobs belong to the key's project).

    # 1. collect pages 400..end of the book and submit them as one batch job
    python -m tools.gemini_batch_lyrics submit Moosburger_Graduale --from 400 --key-dir ~/secrets/gemini --out-dir /data/moosburger_lyrics

    # 2. ask the API how far the job is (batch jobs run up to 24 h)
    python -m tools.gemini_batch_lyrics status --out-dir /data/moosburger_lyrics

    # 3. download and parse the results once the job succeeded
    python -m tools.gemini_batch_lyrics fetch --out-dir /data/moosburger_lyrics

    # 4. optional: submit the pages that produced no result again
    python -m tools.gemini_batch_lyrics resubmit --out-dir /data/moosburger_lyrics

    # 5. optional: insert the transcriptions into the PcGts files of the book
    python -m tools.gemini_batch_lyrics apply --out-dir /data/moosburger_lyrics --write

Only the sung lyric lines are requested, no music and no paratext (rubrics,
headings, folio numbers, ...). The complete page image is sent, so the model
can use the context of the whole page while reading a difficult line, and it
must answer with one entry per *physically written line* in reading order.
No bounding boxes are requested or used: the answer is aligned back onto the
lyric line regions of the PcGts file by that reading order.

Requires the ``google-genai`` package (``uv pip install google-genai``), which
is an optional dependency of the server, just like for the interactive Gemini
text predictor in ``omr/steps/text/llm``.
"""
import argparse
import base64
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

if __name__ == '__main__':
    import django

    # 'python tools/gemini_batch_lyrics.py' puts tools/ on the path instead of
    # the server root, so ommr4all.settings/database would not be importable
    server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if server_root not in sys.path:
        sys.path.insert(0, server_root)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    django.setup()

from PIL import Image

from database import DatabaseBook, DatabasePage

logger = logging.getLogger(__name__)

# gemini-3.1-pro-preview with a high thinking level is the strongest setup for
# heavily abbreviated medieval hands; --model switches to a cheaper one (e.g.
# gemini-3.1-flash-lite), --thinking-level to a cheaper reasoning budget.
DEFAULT_MODEL = 'gemini-3.1-pro-preview'
DEFAULT_THINKING_LEVEL = 'high'
# the high resolution colour scan (2500 px wide), the same image the editor shows
DEFAULT_IMAGE = 'color_highres_preproc'
STATE_FILE_NAME = 'batch_state.json'
STATE_VERSION = 1

# file names looked for inside --key-dir, in this order
API_KEY_FILE_CANDIDATES = [
    'gemini_api_key', 'gemini_api_key.txt', 'gemini.key', 'GEMINI_API_KEY',
    'google_api_key', 'google_api_key.txt', 'GOOGLE_API_KEY',
    'api_key', 'api_key.txt', 'key', 'key.txt',
]
API_KEY_ENV_NAMES = ['GEMINI_API_KEY', 'GOOGLE_API_KEY']

JOB_STATES_DONE = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
                   'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

MIME_TYPES = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}

PROMPT_HEAD = """\
You are an expert palaeographer transcribing a page of a medieval liturgical chant manuscript.
The image shows one complete page: staves of music notation with the sung text (the lyrics) written underneath the notes.

Transcribe ONLY THE SUNG LYRIC LINES, that is the text belonging to a stave.
Ignore everything else on the page: the music notation itself, rubrics and performance instructions (often written in red), headings and titles, folio or page numbers, running heads, marginalia, catchwords, later annotations and stand-alone decorated initials that carry no text line.

Rules:
1. One entry per physically written line. Never merge two written lines into one entry and never split one written line into two entries: a line break in the manuscript is an entry boundary.
2. Reading order: top to bottom, and if the page has columns, the complete left column before the right one.
3. Keep the spelling exactly as written. Do not modernise it, do not expand abbreviations, do not add punctuation that is not in the manuscript.
4. If the syllables of a word are spread apart under the notes, write them separated by single spaces, exactly as they stand in the image.
5. A decorated or enlarged initial belongs to the line it starts: give it as the first letter of that line's text.
6. Transcribe what you can read; leave illegible parts out instead of guessing.
7. You see the whole page, so use its context - the neighbouring lines, the recurring formulas of the chant and the habits of this scribe - to read damaged or heavily abbreviated passages.
"""

PROMPT_TAIL = """\
Answer with a JSON array of strings only, no markdown and no explanation: one string per lyric line, in reading order.

["<text of the first lyric line>", "<text of the second lyric line>", ...]

The entries are matched to the lines of the page by their order, so the number of entries must equal the number of written lyric lines: use an empty string for a lyric line you cannot read at all instead of dropping it.
"""

PROMPT_LINE_COUNT = """\
The layout analysis of this page found {count} lyric line(s). Treat that as a strong hint for the expected number of entries, but follow the image if it clearly shows a different number of written lyric lines.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# api key
# ---------------------------------------------------------------------------
def read_api_key_file(path: str) -> str:
    """Read an API key from a file: a bare key, a ``KEY=value`` env style file
    or a JSON object with a key/api_key/GEMINI_API_KEY field."""
    with open(path) as f:
        content = f.read().strip()
    if not content:
        raise ValueError("The API key file '{}' is empty".format(path))

    if content.startswith('{'):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}
        for name in ['api_key', 'key', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'gemini_api_key']:
            if isinstance(data.get(name), str) and data[name].strip():
                return data[name].strip()

    if '\n' in content or '=' in content:
        for row in content.splitlines():
            row = row.strip().removeprefix('export ').strip()
            if '=' not in row:
                continue
            name, _, value = row.partition('=')
            if name.strip() in API_KEY_ENV_NAMES:
                return value.strip().strip('"\'')

    first = content.splitlines()[0].strip()
    if '=' in first or ' ' in first:
        raise ValueError("Cannot read an API key out of '{}'".format(path))
    return first


def resolve_api_key(key_dir: Optional[str]) -> str:
    """The API key, taken from --key-dir (a directory holding a key file, or the
    key file itself) and falling back to the GEMINI_API_KEY/GOOGLE_API_KEY
    environment variables."""
    key_dir = key_dir or os.environ.get('GEMINI_API_KEY_DIR') or os.environ.get('OMMR4ALL_API_KEY_DIR')
    if key_dir:
        path = os.path.expanduser(key_dir)
        if os.path.isfile(path):
            return read_api_key_file(path)
        if not os.path.isdir(path):
            raise ValueError("The API key path '{}' does not exist".format(path))

        for name in API_KEY_FILE_CANDIDATES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return read_api_key_file(candidate)

        # no conventionally named file: accept the directory if it holds exactly
        # one small non hidden file
        files = [f for f in sorted(os.listdir(path))
                 if not f.startswith('.') and os.path.isfile(os.path.join(path, f))
                 and os.path.getsize(os.path.join(path, f)) < 4096]
        if len(files) == 1:
            return read_api_key_file(os.path.join(path, files[0]))
        raise ValueError(
            "No API key file found in '{}'. Expected one of {} (or a single file holding the key)."
            .format(path, ', '.join(API_KEY_FILE_CANDIDATES[:4])))

    for name in API_KEY_ENV_NAMES:
        if os.environ.get(name):
            return os.environ[name].strip()
    raise ValueError(
        "No API key. Pass --key-dir <folder holding the key file> or set one of {}."
        .format(', '.join(API_KEY_ENV_NAMES)))


# ---------------------------------------------------------------------------
# page selection
# ---------------------------------------------------------------------------
def page_number(page_name: str) -> Optional[int]:
    """The number of a page folder name: the last group of digits, so both
    '007' and 'bav80000018_00400' resolve to 7 and 400."""
    groups = re.findall(r'\d+', page_name)
    return int(groups[-1]) if groups else None


def resolve_page_index(book: DatabaseBook, names: List[str], value: str, is_first: bool) -> int:
    """Index into ``names`` of the page named/numbered ``value``."""
    if value in names:
        return names.index(value)

    if re.fullmatch(r'\d+', value.strip()):
        wanted = int(value)
        numbers = [page_number(n) for n in names]
        for i, number in enumerate(numbers):
            if number == wanted:
                return i
        # no page carries exactly that number: take the closest one inside the
        # range, so --from 400 still works if page 400 is missing
        candidates = [i for i, n in enumerate(numbers)
                      if n is not None and (n >= wanted if is_first else n <= wanted)]
        if candidates:
            i = min(candidates) if is_first else max(candidates)
            logger.warning("Book '%s' has no page %d, using '%s' instead", book.book, wanted, names[i])
            return i

    raise ValueError("Cannot resolve page '{}' in book '{}' ({} pages, '{}' .. '{}')"
                     .format(value, book.book, len(names), names[0] if names else '-',
                             names[-1] if names else '-'))


def select_pages(book: DatabaseBook, first: Optional[str], last: Optional[str]) -> List[DatabasePage]:
    names = book.page_names()
    if not names:
        raise ValueError("Book '{}' has no pages".format(book.book))
    start = resolve_page_index(book, names, first, True) if first else 0
    end = resolve_page_index(book, names, last, False) if last else len(names) - 1
    if end < start:
        raise ValueError("Empty page range: '{}' lies before '{}'".format(last, first))
    return [DatabasePage(book, name) for name in names[start:end + 1]]


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------
def encode_image(path: str, max_width: int, jpeg_quality: int) -> Tuple[str, str, Tuple[int, int]]:
    """base64 payload, mime type and pixel size of the image to send.

    Without --max-width/--jpeg-quality the stored file is sent byte for byte,
    which avoids a second lossy JPEG generation.
    """
    with Image.open(path) as image:
        size = image.size
        if (max_width <= 0 or size[0] <= max_width) and jpeg_quality <= 0:
            with open(path, 'rb') as f:
                data = f.read()
            mime = MIME_TYPES.get(os.path.splitext(path)[1].lower(), 'image/jpeg')
            return base64.b64encode(data).decode('ascii'), mime, size

        image = image.convert('RGB')
        if 0 < max_width < image.width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.LANCZOS)
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=jpeg_quality if jpeg_quality > 0 else 90)
        return base64.b64encode(buffer.getvalue()).decode('ascii'), 'image/jpeg', image.size


def lines_in_reading_order(lines: List) -> List:
    """The lyric lines column by column, top to bottom, and left to right within
    one written row.

    ``sort_lines_in_reading_order`` orders a column by the top edge of a line
    only. That flips the two halves of a written line that a drop capital split
    in the middle whenever the right half sits a few pixels higher, so lines
    that overlap vertically are regrouped into a row and sorted by their left
    edge here.
    """
    from omr.steps.text.llm.predictor import sort_lines_in_reading_order

    ordered, result, row = sort_lines_in_reading_order(lines), [], []

    def flush():
        result.extend(sorted(row, key=lambda l: l.aabb.left()))
        del row[:]

    for line in ordered:
        if row:
            first, current = row[0].aabb, line.aabb
            overlap = min(first.bottom(), current.bottom()) - max(first.top(), current.top())
            if overlap <= 0.5 * min(first.height(), current.height()):
                flush()
        row.append(line)
    flush()
    return result


def lyric_line_count(db_page: DatabasePage) -> int:
    """The number of lyric line regions the layout analysis found on the page,
    0 if it did not run yet."""
    try:
        return len(db_page.pcgts().page.all_text_lines(only_lyric=True))
    except Exception as e:
        logger.debug("Page %s: cannot count lyric lines: %s: %s", db_page.page, type(e).__name__, e)
        return 0


def build_prompt(n_lines: int) -> str:
    parts = [PROMPT_HEAD]
    if n_lines > 0:
        parts.append(PROMPT_LINE_COUNT.replace('{count}', str(n_lines)))
    parts.append(PROMPT_TAIL)
    return '\n'.join(parts)


def build_request(db_page: DatabasePage, args) -> Tuple[Dict, Dict]:
    """The batch request for one page and the metadata needed to interpret its
    answer later on (page name, sent image size, expected number of lines)."""
    image_path = db_page.file(args.image, create_if_not_existing=True).local_path()
    data, mime, size = encode_image(image_path, args.max_width, args.jpeg_quality)
    n_lines = lyric_line_count(db_page)

    generation_config = {'temperature': 0, 'response_mime_type': 'application/json'}
    if args.thinking_level:
        # the REST enum is upper case, the flag is not
        generation_config['thinking_config'] = {'thinking_level': args.thinking_level.upper()}

    request = {
        'key': 'page-' + db_page.page,
        'request': {
            'contents': [{
                'role': 'user',
                'parts': [
                    {'text': build_prompt(n_lines)},
                    {'inline_data': {'mime_type': mime, 'data': data}},
                ],
            }],
            'generation_config': generation_config,
        },
    }
    meta = {'page': db_page.page,
            'image_width': size[0],
            'image_height': size[1],
            'n_expected_lines': n_lines,
            }
    return request, meta


def write_jsonl(requests: List[Dict], path: str) -> int:
    with open(path, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# state file
# ---------------------------------------------------------------------------
class BatchState:
    """The resumable description of one export: which pages were sent in which
    batch job, and which of them already produced a result."""

    def __init__(self, path: str, data: Dict):
        self.path = path
        self.data = data

    @staticmethod
    def state_path(out_dir: str) -> str:
        return os.path.join(out_dir, STATE_FILE_NAME)

    @classmethod
    def create(cls, out_dir: str, book: str, args) -> 'BatchState':
        return cls(cls.state_path(out_dir), {
            'version': STATE_VERSION,
            'created': utc_now(),
            'book': book,
            'model': args.model,
            'thinking_level': args.thinking_level,
            'image': args.image,
            'max_width': args.max_width,
            'jpeg_quality': args.jpeg_quality,
            'out_dir': os.path.abspath(out_dir),
            'jobs': [],
        })

    @classmethod
    def load(cls, out_dir: str) -> 'BatchState':
        out_dir = os.path.abspath(os.path.expanduser(out_dir))
        path = cls.state_path(out_dir)
        if not os.path.isfile(path):
            raise ValueError("No state file at '{}'. Run 'submit' first.".format(path))
        with open(path) as f:
            data = json.load(f)
        if data.get('version') != STATE_VERSION:
            raise ValueError("State file '{}' has unsupported version {}".format(path, data.get('version')))
        # the export directory may have been moved or copied to another machine:
        # where the state file actually lies wins over the path recorded at
        # submit time, so the whole export stays self contained
        if data.get('out_dir') != out_dir:
            logger.info("Export directory moved from %s to %s", data.get('out_dir'), out_dir)
            data['out_dir'] = out_dir
        return cls(path, data)

    def save(self) -> None:
        # write via a temporary file: a crash while saving must not destroy the
        # only record of the running (and paid for) batch jobs
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    @property
    def jobs(self) -> List[Dict]:
        return self.data['jobs']

    @property
    def out_dir(self) -> str:
        return self.data['out_dir']

    def input_path(self, job: Dict) -> str:
        """The request file of ``job``, resolved inside the current export
        directory. Older state files hold an absolute path here."""
        name = job['input_jsonl']
        if os.path.isabs(name):
            if os.path.isfile(name):
                return name
            name = os.path.basename(name)
        return os.path.join(self.out_dir, name)

    def add_job(self, pages: List[Dict], jsonl_path: str) -> Dict:
        job = {'index': len(self.jobs),
               'created': utc_now(),
               # relative to out_dir, so the export survives being moved
               'input_jsonl': os.path.basename(jsonl_path),
               'uploaded_file': None,
               'job_name': None,
               'state': None,
               'submitted': None,
               'result_file': None,
               'fetched': None,
               'pages': pages,
               }
        self.jobs.append(job)
        return job

    def page_metas(self) -> Dict[str, Dict]:
        """page name -> its newest request metadata."""
        metas = {}
        for job in self.jobs:
            for page in job['pages']:
                metas[page['page']] = page
        return metas

    def pages_without_result(self) -> List[str]:
        return [name for name, meta in sorted(self.page_metas().items()) if not meta.get('result')]


# ---------------------------------------------------------------------------
# gemini api
# ---------------------------------------------------------------------------
def genai_client(api_key: str):
    try:
        from google import genai
    except ImportError:
        raise SystemExit("The google-genai package is required for this tool. "
                         "Install it with 'uv pip install google-genai'.")
    return genai.Client(api_key=api_key)


def upload_and_create_job(client, job: Dict, state: BatchState, display_name: str) -> None:
    """Upload the request file of ``job`` and start the batch job for it. Both
    steps are recorded in the state file as they happen, so an interrupted
    submit can be continued without building or paying for anything twice."""
    from google.genai import types

    if not job['uploaded_file']:
        input_path = state.input_path(job)
        logger.info("Uploading %s (%.1f MiB)", input_path,
                    os.path.getsize(input_path) / 1024 / 1024)
        uploaded = client.files.upload(
            file=input_path,
            config=types.UploadFileConfig(display_name=display_name, mime_type='jsonl'))
        job['uploaded_file'] = uploaded.name
        state.save()

    if not job['job_name']:
        batch_job = client.batches.create(model=state.data['model'],
                                          src=job['uploaded_file'],
                                          config={'display_name': display_name})
        job['job_name'] = batch_job.name
        job['state'] = job_state_name(batch_job)
        job['submitted'] = utc_now()
        state.save()
        logger.info("Batch job %s created for %d pages (state %s)",
                    job['job_name'], len(job['pages']), job['state'])


def job_state_name(batch_job) -> str:
    state = getattr(batch_job, 'state', None)
    return getattr(state, 'name', None) or str(state)


def refresh_job(client, job: Dict, state: BatchState) -> Optional[str]:
    if not job['job_name']:
        return None
    batch_job = client.batches.get(name=job['job_name'])
    job['state'] = job_state_name(batch_job)
    dest = getattr(batch_job, 'dest', None)
    if dest is not None and getattr(dest, 'file_name', None):
        job['result_file'] = dest.file_name
    error = getattr(batch_job, 'error', None)
    if error is not None:
        job['error'] = str(error)
    state.save()
    return job['state']


# ---------------------------------------------------------------------------
# result parsing
# ---------------------------------------------------------------------------
def response_text(response: Dict) -> str:
    """The text of a GenerateContentResponse as it appears in the result JSONL."""
    parts = []
    for candidate in response.get('candidates') or []:
        for part in (candidate.get('content') or {}).get('parts') or []:
            if isinstance(part.get('text'), str):
                parts.append(part['text'])
        if parts:
            break
    return ''.join(parts)


def parse_entries(raw: str, image_size: Tuple[int, int]) -> List[Dict]:
    """The transcribed lines of one page, in the order the model returned them,
    as ``{'text': ...}`` dicts.

    Primary format is the requested JSON array of strings; an array of objects
    with a ``text`` field is accepted as well, and anything else is handed to
    the tolerant parser of the interactive LLM predictor, which also copes with
    markdown fences and plain text answers.
    """
    text = raw.strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    data = None
    start = text.find('[')
    if start >= 0:
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            data = None

    if isinstance(data, list) and any(isinstance(e, (str, dict)) for e in data):
        entries = []
        for entry in data:
            if isinstance(entry, str):
                entries.append({'text': entry.strip()})
            elif isinstance(entry, dict):
                entries.append({'text': str(entry.get('text', '') or '').strip()})
        return entries

    from omr.steps.text.llm.adapters import parse_llm_response

    return [{'text': line.text} for line in parse_llm_response(raw, image_size)]


def write_page_result(out_dir: str, state: BatchState, meta: Dict, raw: str) -> Dict:
    """Store raw answer, parsed JSON and plain text of one page. The lines are
    kept in the order the model returned them, which is the order they are
    aligned onto the lyric line regions by."""
    pages_dir = os.path.join(out_dir, 'pages')
    raw_dir = os.path.join(out_dir, 'raw')
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    size = (meta['image_width'], meta['image_height'])
    entries = parse_entries(raw, size)
    expected = meta.get('n_expected_lines') or 0
    if expected and len(entries) != expected:
        logger.warning("Page %s: %d transcribed line(s) but %d lyric line region(s) on the page",
                       meta['page'], len(entries), expected)

    with open(os.path.join(raw_dir, meta['page'] + '.txt'), 'w') as f:
        f.write(raw)

    document = {'book': state.data['book'],
                'page': meta['page'],
                'model': state.data['model'],
                'image': state.data['image'],
                'image_width': size[0],
                'image_height': size[1],
                'n_expected_lines': expected,
                'lines': entries,
                }
    with open(os.path.join(pages_dir, meta['page'] + '.json'), 'w') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    # one physical manuscript line per text line, in reading order
    with open(os.path.join(pages_dir, meta['page'] + '.txt'), 'w') as f:
        f.write('\n'.join(entry['text'] for entry in entries) + '\n')

    return {'n_lines': len(entries), 'fetched': utc_now()}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def collect_requests(pages: List[DatabasePage], args) -> Tuple[List[Dict], List[Dict]]:
    requests, metas = [], []
    for i, db_page in enumerate(pages):
        try:
            request, meta = build_request(db_page, args)
        except Exception as e:
            # a single broken page must not cost the whole batch
            logger.error("Skipping page %s: %s: %s", db_page.page, type(e).__name__, e)
            continue
        requests.append(request)
        metas.append(meta)
        if (i + 1) % 25 == 0 or i + 1 == len(pages):
            logger.info("Prepared %d/%d pages", i + 1, len(pages))
    return requests, metas


def submit_requests(state: BatchState, requests: List[Dict], metas: List[Dict],
                    out_dir: str, args) -> None:
    """Chunk, write, upload and start; every stage is recorded in the state."""
    chunk = args.max_pages_per_job if args.max_pages_per_job > 0 else len(requests)
    n_jobs_before = len(state.jobs)
    for offset in range(0, len(requests), chunk):
        part_requests = requests[offset:offset + chunk]
        part_metas = metas[offset:offset + chunk]
        jsonl_path = os.path.join(out_dir, 'requests_{:03d}.jsonl'.format(len(state.jobs)))
        size = write_jsonl(part_requests, jsonl_path)
        logger.info("Wrote %d requests to %s (%.1f MiB)", len(part_requests), jsonl_path,
                    size / 1024 / 1024)
        if size > 1900 * 1024 * 1024:
            logger.warning("The request file exceeds the 2 GB limit of the file API. "
                           "Use --max-pages-per-job or --max-width to make it smaller.")
        state.add_job(part_metas, jsonl_path)

    if args.dry_run:
        # forget the jobs again: a state that lists unsubmitted jobs would make
        # the next real submit upload these files as well
        n_written = len(state.jobs) - n_jobs_before
        del state.jobs[n_jobs_before:]
        if n_jobs_before:
            state.save()
        print("Dry run: {} request file(s) written to {}, nothing uploaded."
              .format(n_written, os.path.abspath(out_dir)))
        return
    state.save()

    client = genai_client(resolve_api_key(args.key_dir))
    for job in state.jobs:
        if job['job_name']:
            continue
        upload_and_create_job(client, job, state,
                              '{}-lyrics-{:03d}'.format(state.data['book'], job['index']))
    print_status(state)


def cmd_submit(args) -> None:
    book = DatabaseBook(args.book)
    if not book.exists():
        raise ValueError("Book '{}' does not exist. Available: {}".format(
            args.book, ', '.join(b.book for b in DatabaseBook.list_available())))

    out_dir = args.out_dir or os.path.join(os.getcwd(), 'gemini_lyrics_' + book.book)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(BatchState.state_path(out_dir)) and not args.force:
        raise ValueError("'{}' already holds a batch state. Use 'status'/'fetch'/'resubmit' to "
                         "continue it, --out-dir for a new export or --force to overwrite it."
                         .format(os.path.abspath(out_dir)))

    pages = select_pages(book, args.from_page, args.to_page)
    logger.info("Book %s: %d pages selected (%s .. %s)", book.book, len(pages),
                pages[0].page, pages[-1].page)

    # the state file is written by submit_requests once there is something to
    # remember, so that a --dry-run leaves the directory reusable
    state = BatchState.create(out_dir, book.book, args)
    requests, metas = collect_requests(pages, args)
    if not requests:
        raise ValueError("No page could be prepared, nothing to submit")
    submit_requests(state, requests, metas, out_dir, args)


def cmd_resubmit(args) -> None:
    state = BatchState.load(args.out_dir)
    missing = state.pages_without_result()
    if not missing:
        print("Every page of this export has a result, nothing to resubmit.")
        return

    book = DatabaseBook(state.data['book'])
    # reuse the settings of the original submit so the results stay comparable
    args.image = state.data['image']
    args.max_width = state.data['max_width']
    args.jpeg_quality = state.data['jpeg_quality']
    if args.thinking_level is None:
        args.thinking_level = state.data.get('thinking_level')
    logger.info("Resubmitting %d page(s) without result", len(missing))
    requests, metas = collect_requests([DatabasePage(book, name) for name in missing], args)
    if not requests:
        raise ValueError("No page could be prepared, nothing to submit")
    submit_requests(state, requests, metas, state.out_dir, args)


def print_status(state: BatchState) -> None:
    metas = state.page_metas()
    done = [name for name, meta in metas.items() if meta.get('result')]
    print("Export of book '{}' in {}".format(state.data['book'], state.out_dir))
    print("  model {} (thinking {}), image {}, {} page(s), {} with result".format(
        state.data['model'], state.data.get('thinking_level') or 'default',
        state.data['image'], len(metas), len(done)))
    for job in state.jobs:
        print("  job {:03d}: {} pages, state {}, name {}{}".format(
            job['index'], len(job['pages']), job['state'] or 'not submitted',
            job['job_name'] or '-',
            ', error: ' + job['error'] if job.get('error') else ''))


def cmd_status(args) -> None:
    state = BatchState.load(args.out_dir)
    client = genai_client(resolve_api_key(args.key_dir))
    for job in state.jobs:
        refresh_job(client, job, state)
    print_status(state)


def cmd_fetch(args) -> None:
    state = BatchState.load(args.out_dir)
    out_dir = state.out_dir
    client = genai_client(resolve_api_key(args.key_dir))

    for job in state.jobs:
        job_state = refresh_job(client, job, state)
        if job_state != 'JOB_STATE_SUCCEEDED':
            logger.info("Job %03d is %s, skipping", job['index'], job_state or 'not submitted')
            continue
        if job['fetched'] and not args.force:
            logger.info("Job %03d was already fetched at %s, skipping (--force to redo)",
                        job['index'], job['fetched'])
            continue
        if not job['result_file']:
            logger.warning("Job %03d succeeded but reports no result file", job['index'])
            continue

        logger.info("Downloading results of job %03d (%s)", job['index'], job['result_file'])
        content = client.files.download(file=job['result_file'])
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        result_path = os.path.join(out_dir, 'results_{:03d}.jsonl'.format(job['index']))
        with open(result_path, 'w') as f:
            f.write(content)

        metas = {'page-' + page['page']: page for page in job['pages']}
        by_position = list(job['pages'])
        n_ok, n_error = 0, 0
        for i, row in enumerate(content.splitlines()):
            row = row.strip()
            if not row:
                continue
            try:
                data = json.loads(row)
            except json.JSONDecodeError:
                logger.error("Result line %d of job %03d is not valid JSON", i, job['index'])
                continue
            meta = metas.get(data.get('key'))
            if meta is None and i < len(by_position):
                # older/other API versions may not echo the key back
                meta = by_position[i]
            if meta is None:
                logger.error("Cannot map result with key %r to a page", data.get('key'))
                continue

            if data.get('error'):
                meta['error'] = json.dumps(data['error'])[:500]
                logger.error("Page %s failed: %s", meta['page'], meta['error'])
                n_error += 1
                continue
            raw = response_text(data.get('response') or {})
            if not raw.strip():
                meta['error'] = 'empty response'
                logger.error("Page %s returned an empty response", meta['page'])
                n_error += 1
                continue
            meta.pop('error', None)
            meta['result'] = write_page_result(out_dir, state, meta, raw)
            n_ok += 1

        job['fetched'] = utc_now()
        state.save()
        logger.info("Job %03d: %d page(s) written, %d failed", job['index'], n_ok, n_error)

    print_status(state)
    missing = state.pages_without_result()
    if missing:
        print("  {} page(s) without result: {}{}".format(
            len(missing), ', '.join(missing[:10]), ' ...' if len(missing) > 10 else ''))
        print("  run 'resubmit' to send them again")


def cmd_cancel(args) -> None:
    state = BatchState.load(args.out_dir)
    client = genai_client(resolve_api_key(args.key_dir))
    for job in state.jobs:
        if not job['job_name'] or job['state'] in JOB_STATES_DONE:
            continue
        logger.info("Cancelling job %03d (%s)", job['index'], job['job_name'])
        client.batches.cancel(name=job['job_name'])
        refresh_job(client, job, state)
    print_status(state)


def cmd_apply(args) -> None:
    """Insert the fetched transcriptions into the PcGts files of the book.

    The model answered with one entry per written lyric line in reading order,
    so the entries are zipped onto the lyric line regions of the page in the
    same order.
    """
    from database.file_formats.pcgts.page import Sentence
    from omr.steps.text.llm.predictor import join_spaced_syllables
    from omr.steps.text.hyphenation.hyphenator import CombinedHyphenator, HyphenDicts

    state = BatchState.load(args.out_dir)
    book = DatabaseBook(state.data['book'])
    hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1, right=1)

    pages_dir = os.path.join(state.out_dir, 'pages')
    for name in sorted(state.page_metas()):
        result_path = os.path.join(pages_dir, name + '.json')
        if not os.path.isfile(result_path):
            continue
        with open(result_path) as f:
            document = json.load(f)

        db_page = DatabasePage(book, name)
        pcgts = db_page.pcgts()
        page = pcgts.page
        target_lines = lines_in_reading_order(page.all_text_lines(only_lyric=True))
        if not target_lines:
            logger.warning("Page %s has no lyric line regions, skipping", name)
            continue

        texts = [str(e.get('text') or '').strip() for e in document['lines']]
        if len(texts) != len(target_lines):
            logger.warning("Page %s: %d transcribed line(s) for %d lyric line region(s). Aligning "
                           "in reading order, the surplus is dropped.",
                           name, len(texts), len(target_lines))
        assignment = {line.id: text for line, text in zip(target_lines, texts)}

        changed = 0
        for line in target_lines:
            text = assignment.get(line.id, '')
            if not text:
                continue
            text = join_spaced_syllables(text, hyphen.dictionary)
            hyphenated = hyphen.apply_to_sentence(text)
            if args.write:
                line.sentence = Sentence.from_string(hyphenated)
            else:
                print('{} {}: {}'.format(name, line.id, hyphenated))
            changed += 1

        if args.write and changed:
            page.annotations.connections.clear()
            pcgts.to_file(db_page.file('pcgts').local_path())
        logger.info("Page %s: %d line(s) %s", name, changed,
                    'written' if args.write else 'would be written')

    if not args.write:
        print("Preview only, nothing was written. Add --write to store the lyrics in the PcGts files.")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def add_key_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--key-dir', type=str, default=None,
                        help='folder holding the Gemini API key file (or the key file itself). '
                             'Defaults to $GEMINI_API_KEY_DIR, then $GEMINI_API_KEY/$GOOGLE_API_KEY.')


def add_out_dir_argument(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument('--out-dir', type=str, required=required, default=None,
                        help='directory holding the requests, the state file and the results')


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # the indented block of the module docstring holds the usage examples
        epilog='\n'.join(row for row in __doc__.splitlines()
                         if row.startswith('    ') or not row.strip()).strip('\n'))
    parser.add_argument('--verbose', action='store_true', help='debug logging')
    commands = parser.add_subparsers(dest='command', required=True)

    submit = commands.add_parser('submit', help='collect a page range and start the batch job(s)')
    submit.add_argument('book', type=str, help='name of the book, e.g. Moosburger_Graduale')
    submit.add_argument('--from', dest='from_page', type=str, default=None,
                        help='first page, as page name or page number (default: first page of the book)')
    submit.add_argument('--to', dest='to_page', type=str, default=None,
                        help='last page, as page name or page number (default: last page of the book)')
    add_out_dir_argument(submit, required=False)
    add_key_dir_argument(submit)
    submit.add_argument('--model', type=str, default=os.environ.get('GEMINI_MODEL') or DEFAULT_MODEL,
                        help='Gemini model id (default: %(default)s)')
    submit.add_argument('--image', type=str, default=DEFAULT_IMAGE,
                        help='page image to send, the complete page (default: %(default)s)')
    submit.add_argument('--max-width', type=int, default=0,
                        help='downscale the image to this width before sending (0: send it unchanged)')
    submit.add_argument('--jpeg-quality', type=int, default=0,
                        help='re-encode the image with this JPEG quality (0: send the stored file as is)')
    submit.add_argument('--thinking-level', type=str, default=DEFAULT_THINKING_LEVEL,
                        choices=['low', 'high'],
                        help='reasoning level for Gemini 3 models (default: %(default)s)')
    submit.add_argument('--max-pages-per-job', type=int, default=0,
                        help='split the export into several batch jobs of at most this many pages')
    submit.add_argument('--dry-run', action='store_true',
                        help='only build the request file(s), do not upload anything')
    submit.add_argument('--force', action='store_true', help='overwrite an existing state file')

    status = commands.add_parser('status', help='poll the batch job(s) of an export')
    add_out_dir_argument(status, required=True)
    add_key_dir_argument(status)

    fetch = commands.add_parser('fetch', help='download and parse the results of finished job(s)')
    add_out_dir_argument(fetch, required=True)
    add_key_dir_argument(fetch)
    fetch.add_argument('--force', action='store_true', help='fetch jobs again that were fetched before')

    resubmit = commands.add_parser('resubmit', help='submit the pages that produced no result again')
    add_out_dir_argument(resubmit, required=True)
    add_key_dir_argument(resubmit)
    resubmit.add_argument('--max-pages-per-job', type=int, default=0,
                          help='split into several batch jobs of at most this many pages')
    resubmit.add_argument('--thinking-level', type=str, default=None, choices=['low', 'high'],
                          help='reasoning level (default: the one of the original submit)')
    resubmit.add_argument('--dry-run', action='store_true',
                          help='only build the request file, do not upload anything')

    cancel = commands.add_parser('cancel', help='cancel the running batch job(s) of an export')
    add_out_dir_argument(cancel, required=True)
    add_key_dir_argument(cancel)

    apply_ = commands.add_parser('apply', help='insert the fetched lyrics into the PcGts of the book')
    add_out_dir_argument(apply_, required=True)
    apply_.add_argument('--write', action='store_true',
                        help='actually store the lyrics (without it only a preview is printed)')

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    # settings.LOGGING already gives '__main__' a console handler; adding one via
    # basicConfig on top of it would print every message twice
    if not logger.handlers and not logging.getLogger().handlers:
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
    logger.setLevel(level)
    commands = {'submit': cmd_submit, 'status': cmd_status, 'fetch': cmd_fetch,
                'resubmit': cmd_resubmit, 'cancel': cmd_cancel, 'apply': cmd_apply}
    try:
        commands[args.command](args)
    except ValueError as e:
        print('error: {}'.format(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
