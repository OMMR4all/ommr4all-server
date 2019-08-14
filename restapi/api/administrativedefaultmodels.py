from rest_framework.views import APIView
from .auth import require_global_permissions, DatabasePermissionFlag
from rest_framework.response import Response
from omr.steps.step import Step
from omr.steps.algorithmtypes import AlgorithmGroups
from database.model import ModelMeta, Model
from database.database_available_models import DatabaseAvailableModels
import logging
logger = logging.getLogger(__name__)


class AdministrativeDefaultModelsView(APIView):
    @require_global_permissions([DatabasePermissionFlag.CHANGE_DEFAULT_MODEL_FOR_BOOK_STYLE])
    def put(self, request, group, style):
        meta = ModelMeta.from_json(request.body)
        default_type = AlgorithmGroups(group).types()[0]
        model = Model.from_id_str(meta.id)
        target_model = Model(DatabaseAvailableModels.local_default_model_path_for_style(style, Step.meta(default_type).model_dir()))
        model.copy_to(target_model, override=True)
        return Response()

    def get(self, request, group, style):
        default_type = AlgorithmGroups(group).types()[0]
        return Response(Step.meta(default_type).list_available_models_for_style(style).to_dict())
