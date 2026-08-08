from typing import List, Optional, Tuple

from rest_framework.views import APIView
from .auth import require_global_permissions, DatabasePermissionFlag
from rest_framework.response import Response
from rest_framework import permissions, status
from omr.steps.step import Step
from omr.steps.algorithmtypes import AlgorithmGroups, AlgorithmTypes
from database.model import ModelMeta, Model, MetaId, ModelsId
from database.database_available_models import DatabaseAvailableModels
import logging
logger = logging.getLogger(__name__)


def default_model_slots() -> List[Tuple[AlgorithmTypes, List[AlgorithmTypes]]]:
    """The (algorithm type, aliases) pairs a default model may be configured for, in group order.

    Only model based steps that are registered in omr/steps/step.py are offered: an unregistered
    type has no meta and could not be predicted with anyway. Types that write the same default
    model directory collapse into a single slot, otherwise two rows of the table would overwrite
    each other; the collapsed types are reported as its aliases. The directory comes from the
    step's meta, not from AlgorithmTypes.model_type(): a step may share the trained models of
    another one and still keep an own default (see the torch syllable step, which runs a Guppy
    OCR model but has an own default model directory).
    """
    Step._lazy_load_registry()
    slots = []
    by_model_dir = {}
    for group, types in AlgorithmGroups.group_types_mapping().items():
        for t in types:
            if not t.uses_model() or t not in Step.METAS:
                continue
            model_dir = Step.meta(t).model_dir()
            slot = by_model_dir.get(model_dir)
            if slot is not None:
                slot[1].append(t)
                continue
            by_model_dir[model_dir] = (t, [])
            slots.append((t, by_model_dir[model_dir][1]))
    return slots


def _slot_for(algorithm_type: AlgorithmTypes) -> Optional[Tuple[AlgorithmTypes, List[AlgorithmTypes]]]:
    for slot in default_model_slots():
        if algorithm_type == slot[0] or algorithm_type in slot[1]:
            return slot
    return None


class AdministrativeDefaultModelSlotsView(APIView):
    """The default model slots offered by this server, in the order the client renders them."""
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request):
        return Response({'slots': [{
            'type': t.value,
            'group': t.group().value,
            'model_dir': Step.meta(t).model_dir(),
            'aliases': [a.value for a in aliases],
        } for t, aliases in default_model_slots()]})


class AdministrativeDefaultModelsTypeView(APIView):
    """Read/write the default model of one algorithm for one book style.

    Setting a default physically copies the model into
    ``internal_storage/default_models/<style>/<model_dir>``, from where
    AlgorithmMeta.default_model_for_style picks it up at prediction time.
    """
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @staticmethod
    def _resolve(type: str):
        try:
            algorithm_type = AlgorithmTypes(type)
        except ValueError:
            return None
        slot = _slot_for(algorithm_type)
        return slot[0] if slot else None

    @staticmethod
    def _unknown_type(type: str) -> Response:
        # 400, not 404: a 404 would be caught by APPEND_SLASH and redirected into the webapp
        return Response({'error': "No default model can be set for '{}'".format(type)},
                        status=status.HTTP_400_BAD_REQUEST)

    @require_global_permissions([DatabasePermissionFlag.CHANGE_DEFAULT_MODEL_FOR_BOOK_STYLE])
    def put(self, request, type, style):
        algorithm_type = AdministrativeDefaultModelsTypeView._resolve(type)
        if algorithm_type is None:
            return AdministrativeDefaultModelsTypeView._unknown_type(type)

        meta = ModelMeta.from_json(request.body)
        model = Model.from_id_str(meta.id)
        target_meta = MetaId(DatabaseAvailableModels.local_default_models(style, algorithm_type),
                             Step.meta(algorithm_type).model_dir())
        model.copy_to(Model(target_meta), override=True)
        return Response()

    def get(self, request, type, style):
        algorithm_type = AdministrativeDefaultModelsTypeView._resolve(type)
        if algorithm_type is None:
            return AdministrativeDefaultModelsTypeView._unknown_type(type)

        step = Step.meta(algorithm_type)
        # default_model_for_style falls back to french14, so a model in the response does not
        # imply that this style has one of its own -- the client shows that difference
        own = Model(MetaId(DatabaseAvailableModels.local_default_models(style, algorithm_type),
                           step.model_dir()))
        # has_weights, not exists: internal_storage ships a bare meta.json for every algorithm,
        # which is not a default model anybody could predict with
        return Response({**step.list_available_models_for_style(style).to_dict(),
                         'has_own_default': own.exists() and own.has_weights()})


class AdministrativeDefaultModelsView(APIView):
    """Group scoped default models, kept for older clients.

    A group has no default model of its own; the request is applied to the group's primary
    algorithm (the first entry of AlgorithmGroups.types()).
    """
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @staticmethod
    def _type(group: str) -> str:
        try:
            return AlgorithmGroups(group).types()[0].value
        except ValueError:
            # the client knows groups the server has no algorithms for (documents, search)
            return group

    def put(self, request, group, style):
        return AdministrativeDefaultModelsTypeView().put(
            request, AdministrativeDefaultModelsView._type(group), style)

    def get(self, request, group, style):
        return AdministrativeDefaultModelsTypeView().get(
            request, AdministrativeDefaultModelsView._type(group), style)
