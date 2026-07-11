from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView


class OperationWorkerResourcesView(APIView):
    """Reports, for one operation (an AlgorithmTypes value or a training
    operation name), which worker resources (CPU/GPU) it may run on, which one
    is the default, and the current worker/queue occupancy. The client uses
    this to render the resource selection when starting a task."""
    permission_classes = [permissions.AllowAny]

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
