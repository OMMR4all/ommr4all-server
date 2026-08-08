from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


class OperationWorkerResourcesView(APIView):
    """Reports, for one operation (an AlgorithmTypes value or a training
    operation name), which worker resources (CPU/GPU) it may run on, which one
    is the default, and the current worker/queue occupancy. The client uses
    this to render the resource selection when starting a task.

    Requires authentication (the global default): the worker layout and queue
    occupancy of the server are not public information."""

    def get(self, request, operation):
        from omr.steps.algorithmtypes import AlgorithmTypes, WorkerResource
        from omr.steps.step import Step
        from restapi.operationworker.workerresources import \
            TRAIN_OPERATIONS, groups_for, resolve_worker_resource, n_workers, n_free, n_queued

        training = operation in TRAIN_OPERATIONS
        if training:
            algorithm_type = TRAIN_OPERATIONS[operation]
        else:
            try:
                algorithm_type = AlgorithmTypes(operation)
            except ValueError:
                # 400, not 404: a 404 would be caught by APPEND_SLASH and
                # redirected into the webapp catch-all route
                return Response({'error': "Unknown operation '{}'".format(operation)},
                                status=status.HTTP_400_BAD_REQUEST)

        meta = Step.create_meta(algorithm_type)
        allowed = meta.allowed_trainer_resources() if training else meta.allowed_predictor_resources()
        default = resolve_worker_resource(meta, None, training=training)

        def info(resource: WorkerResource) -> dict:
            groups = groups_for(resource, training=training)
            return {
                'allowed': resource in allowed,
                'default': resource == default,
                'n_workers': n_workers(groups),
                'n_free': n_free(groups),
                'n_tasks_queued': n_queued(groups),
            }

        return Response({
            'operation': operation,
            'resources': {resource.value: info(resource) for resource in WorkerResource},
        })


class OperationTrainParamsView(APIView):
    """The trainer settings a user may choose for one training operation.

    ``n_epoch_max`` is null when the user may train for as long as they like; everybody else is
    capped at the algorithm default. The limit is re-applied when the training is started, this
    endpoint only tells the client what to offer."""

    def get(self, request, operation):
        from database.models.permissions import DatabasePermissionFlag
        from restapi.operationworker.workerresources import TRAIN_OPERATIONS, default_n_epoch
        from restapi.views.auth import is_admin

        if operation not in TRAIN_OPERATIONS:
            # 400, not 404: a 404 would be caught by APPEND_SLASH and redirected into the webapp
            return Response({'error': "Unknown training operation '{}'".format(operation)},
                            status=status.HTTP_400_BAD_REQUEST)

        default = default_n_epoch(TRAIN_OPERATIONS[operation])
        may_increase = is_admin(request.user, DatabasePermissionFlag.SET_TRAINING_EPOCHS)
        return Response({
            'operation': operation,
            'n_epoch_default': default,
            'n_epoch_max': None if may_increase else default,
        })
