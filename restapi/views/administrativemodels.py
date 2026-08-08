"""Administrative overview of every trained model on this server, and pruning.

Models pile up: every training run writes a new directory next to the book
(``<book>/models/<model_dir>/<timestamp>/``), and nothing ever removes them. This view lists
them with their size and their usage (see Model.mark_used) and lets an administrator delete
the ones that are not needed anymore. Models that are still relied on -- the newest one of a
book and step, the one a book explicitly selected, and the ones a notation style's default was
made from -- are protected and are refused even if a client asks for them.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

import ommr4all.settings as settings
from database.database_book import DatabaseBook
from database.model import Model, ModelTrainingInfo, Storage
from database.model.definitions import StorageType
from omr.steps.algorithmtypes import AlgorithmTypes
from .auth import DatabasePermissionFlag, require_admin

logger = logging.getLogger(__name__)

# model directory (an AlgorithmTypes value or a legacy name) and model name (an ISO timestamp
# for trained models, the model dir for the internal defaults)
_NAME_RE = re.compile(r'^[\w\-.:+]+$')

PROTECTION_NEWEST = 'newest'
PROTECTION_SELECTED = 'selected'
PROTECTION_DEFAULT_SOURCE = 'default_source'
PROTECTION_IS_DEFAULT = 'is_default'


def _default_models_root() -> str:
    return os.path.join(Storage(StorageType.INTERNAL).path(), 'default_models')


def _algorithm_types_of(model_dir: str) -> List[str]:
    """The algorithm types served by a model directory (several types may share one).

    Registered steps are matched by the directory their meta names, which is not always the
    model type (the syllable step keeps an own default directory); unregistered and legacy
    directories still resolve through model_type().
    """
    from omr.steps.step import Step
    Step._lazy_load_registry()
    types = [t.value for t, meta in Step.METAS.items() if meta.model_dir() == model_dir]
    if types:
        return types
    return [t.value for t in AlgorithmTypes if t.model_type().value == model_dir]


def _list_dirs(path: str) -> List[str]:
    try:
        with os.scandir(path) as it:
            return sorted([e.name for e in it if e.is_dir()])
    except OSError:
        return []


class ModelAtPath(Model):
    """A model addressed by its path instead of by a parsed MetaId.

    Legacy directories (``pc_staff_lines``) are not valid AlgorithmTypes, so they have no MetaId
    at all -- but they are exactly the leftovers an administrator wants to see and delete.
    """

    def __init__(self, path: str, model_id: str):
        self.meta_id = None
        self.path = path
        self.meta_path = os.path.join(path, Model.META_FILE)
        self.name = os.path.basename(path)
        self._id = model_id
        self._meta = None

    def id(self) -> str:
        return self._id


def _training_summary(info: Optional[ModelTrainingInfo]) -> Optional[dict]:
    """The training record in one line; the full one is served by AdministrativeModelTrainingView."""
    if info is None:
        return None
    return {
        'books': len(info.books),
        'trainPages': info.n_train_pages,
        'validationPages': info.n_validation_pages,
        'nEpoch': info.n_epoch,
        'pretrained': info.pretrained_model,
        'startedBy': info.started_by,
        'finished': info.finished,
    }


def _entry(storage: str, owner: str, model_dir: str, name: str) -> dict:
    path = _model_path(storage, owner, model_dir, name)
    model = ModelAtPath(path, '/'.join([storage, owner, model_dir, name]))
    meta = model.meta()
    usage = model.usage()
    return {
        'id': '/'.join([storage, owner, model_dir, name]),
        'storage': storage,
        'book': owner if storage == StorageType.EXTERNAL.value else None,
        'style': owner if storage == StorageType.INTERNAL.value else meta.style,
        'modelDir': model_dir,
        'algorithmTypes': _algorithm_types_of(model_dir),
        'name': name,
        'created': meta.created,
        'accuracy': meta.accuracy,
        'iters': meta.iters,
        'sourceId': meta.source_id,
        'hasMeta': model.exists(),
        'hasWeights': model.has_weights(),
        'size': model.size(),
        'lastUsed': usage.last_used,
        'nUsed': usage.n_used,
        'training': _training_summary(model.training_info()),
        'protection': [],
    }


def _model_path(storage: str, owner: str, model_dir: str, name: str) -> str:
    if storage == StorageType.EXTERNAL.value:
        return os.path.join(settings.PRIVATE_MEDIA_ROOT, owner, 'models', model_dir, name)
    return os.path.join(_default_models_root(), owner, name)


def parse_model_id(model_id: str) -> Optional[Tuple[str, str, str, str]]:
    """``<storage>/<owner>/<model_dir>/<name>`` -> its parts, or None when it is not addressable.

    Deliberately stricter than MetaId.from_str: the parts end up in a filesystem path, so
    anything that is not a plain name (no separators, no '..') is rejected.
    """
    parts = (model_id or '').split('/')
    if len(parts) != 4:
        return None
    storage, owner, model_dir, name = parts
    if storage not in (StorageType.EXTERNAL.value, StorageType.INTERNAL.value):
        return None
    if not all(_NAME_RE.match(p) for p in (owner, model_dir, name)):
        return None
    path = os.path.realpath(_model_path(storage, owner, model_dir, name))
    root = os.path.realpath(settings.PRIVATE_MEDIA_ROOT if storage == StorageType.EXTERNAL.value
                            else _default_models_root())
    if not path.startswith(root + os.sep) or not os.path.isdir(path):
        return None
    return storage, owner, model_dir, name


def _default_model_entries() -> List[dict]:
    entries = []
    root = _default_models_root()
    for style in _list_dirs(root):
        for model_dir in _list_dirs(os.path.join(root, style)):
            entries.append(_entry(StorageType.INTERNAL.value, style, model_dir, model_dir))
    return entries


def _book_model_entries(books: List[dict]) -> List[dict]:
    entries = []
    for book in books:
        models_root = os.path.join(settings.PRIVATE_MEDIA_ROOT, book['id'], 'models')
        for model_dir in _list_dirs(models_root):
            for name in _list_dirs(os.path.join(models_root, model_dir)):
                entry = _entry(StorageType.EXTERNAL.value, book['id'], model_dir, name)
                entry['bookName'] = book['name']
                entries.append(entry)
    return entries


def _selected_model_ids() -> set:
    """The models books point at explicitly (book meta algorithmPredictorParams[type].modelId)."""
    selected = set()
    for book in DatabaseBook.list_available():
        try:
            params = book.get_meta().algorithmPredictorParams
        except Exception:
            continue
        for p in params.values():
            if p.modelId:
                selected.add(_normalized_meta_id(str(p.modelId)))
    return selected


def _normalized_meta_id(model_id: str) -> str:
    """A MetaId string in the location form used here (storage/owner/model directory/name).

    A stored id may name an algorithm that shares another one's directory (see
    AlgorithmTypes.model_type), so resolve that before comparing against the directories on disk.
    """
    parts = model_id.strip('/').split('/')
    if len(parts) == 4:
        try:
            parts[2] = AlgorithmTypes(parts[2]).model_type().value
        except ValueError:
            pass
    return '/'.join(parts)


def collect_models() -> List[dict]:
    """Every model on disk with its protection reasons resolved."""
    from database.book_index import list_books_synced

    books = [{'id': row.name, 'name': row.display_name or row.name} for row in list_books_synced()]
    defaults = _default_model_entries()
    entries = _book_model_entries(books) + defaults

    # a book model backing a style default: either the default still carries its source id
    # (Model.copy_to), or -- for defaults copied before that -- its own stored id
    protected_by_default = set()
    for d in defaults:
        d['protection'].append(PROTECTION_IS_DEFAULT)
        for candidate in (d['sourceId'], d['id']):
            if candidate:
                protected_by_default.add(_normalized_meta_id(str(candidate)))

    selected = _selected_model_ids()

    # the newest model of a book and step is what predictions fall back on, keep it
    newest: Dict[Tuple[str, str], str] = {}
    for e in entries:
        if e['storage'] != StorageType.EXTERNAL.value or not e['hasMeta']:
            continue
        key = (e['book'], e['modelDir'])
        if key not in newest or e['name'] > newest[key]:
            newest[key] = e['name']

    for e in entries:
        if e['storage'] == StorageType.EXTERNAL.value and newest.get((e['book'], e['modelDir'])) == e['name']:
            e['protection'].append(PROTECTION_NEWEST)
        if e['id'] in selected:
            e['protection'].append(PROTECTION_SELECTED)
        if e['id'] in protected_by_default:
            e['protection'].append(PROTECTION_DEFAULT_SOURCE)

    return sorted(entries, key=lambda e: (e['storage'], e['book'] or e['style'] or '',
                                          e['modelDir'], e['name']))


class AdministrativeModelsView(APIView):
    """All models on this server: where they are, how big, when they were last used."""

    @require_admin(DatabasePermissionFlag.MANAGE_MODELS)
    def get(self, request):
        from restapi.views.bookaccess import etag_response
        models = collect_models()
        return etag_response(request, {
            'models': models,
            'totals': {
                'count': len(models),
                'size': sum(m['size'] for m in models),
            },
        })


class AdministrativeModelTrainingView(APIView):
    """The full training record of one model (its books, pages and hyper parameters).

    Not part of the listing: the page lists of a few hundred models add up.
    """

    @require_admin(DatabasePermissionFlag.MANAGE_MODELS)
    def get(self, request):
        parts = parse_model_id(request.GET.get('id', ''))
        if parts is None:
            return Response({'error': 'Unknown model'}, status=status.HTTP_400_BAD_REQUEST)

        storage, owner, model_dir, name = parts
        model = ModelAtPath(_model_path(storage, owner, model_dir, name), request.GET.get('id'))
        info = model.training_info()
        if info is None:
            # trained before the record existed, or never trained at all
            return Response({'id': model.id(), 'training': None})
        return Response({'id': model.id(), 'training': info.to_dict()})


class AdministrativeModelsPruneView(APIView):
    """Delete the requested models, skipping the protected ones.

    The protection is resolved again here: a stale or hand-crafted request must not be able to
    remove the model a book is about to predict with.
    """

    @require_admin(DatabasePermissionFlag.MANAGE_MODELS)
    def post(self, request):
        try:
            body = json.loads(request.body) if request.body else {}
            ids = list(body.get('ids', []))
        except ValueError:
            return Response({'error': 'Invalid request body'}, status=status.HTTP_400_BAD_REQUEST)

        by_id = {m['id']: m for m in collect_models()}
        deleted, refused, freed = [], [], 0
        for model_id in ids:
            entry = by_id.get(model_id)
            if entry is None or parse_model_id(model_id) is None:
                refused.append({'id': model_id, 'reason': 'unknown'})
                continue
            if entry['protection']:
                refused.append({'id': model_id, 'reason': entry['protection'][0]})
                continue

            storage, owner, model_dir, name = parse_model_id(model_id)
            path = _model_path(storage, owner, model_dir, name)
            try:
                import shutil
                shutil.rmtree(path)
            except OSError as e:
                logger.warning('Could not delete model {}'.format(path))
                logger.exception(e)
                refused.append({'id': model_id, 'reason': 'error'})
                continue
            freed += entry['size']
            deleted.append(model_id)

        return Response({'deleted': deleted, 'refused': refused, 'freed': freed})
