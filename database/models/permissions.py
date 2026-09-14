from django.db import models
from enum import Enum


class DatabasePermissionFlag(Enum):
    ADD_BOOK_STYLE = 'add_book_style'
    DELETE_BOOK_STYLE = 'delete_book_style'
    EDIT_BOOK_STYLE = 'edit_book_style'

    ADD_SYMBOL_CLASS = 'add_symbol_class'
    DELETE_SYMBOL_CLASS = 'delete_symbol_class'
    EDIT_SYMBOL_CLASS = 'edit_symbol_class'

    CHANGE_DEFAULT_MODEL_FOR_BOOK_STYLE = 'change_default_model_for_book_style'

    TASKS_LIST = 'tasks_list'
    TASKS_CANCEL = 'tasks_cancel'

    # administrative flags: these are only ever checked through restapi.views.auth.is_admin,
    # which also accepts Django's is_staff/is_superuser
    SET_TRAINING_EPOCHS = 'set_training_epochs'
    VIEW_SYSTEM_RESOURCES = 'view_system_resources'
    MANAGE_MODELS = 'manage_models'
    MANAGE_TASK_WORKERS = 'manage_task_workers'


class GlobalPermissions(models.Model):
    class Meta:
        permissions = [
            (DatabasePermissionFlag.ADD_BOOK_STYLE.value, 'Add book style'),
            (DatabasePermissionFlag.DELETE_BOOK_STYLE.value, 'Delete book style'),
            (DatabasePermissionFlag.EDIT_BOOK_STYLE.value, 'Edit book style'),
            (DatabasePermissionFlag.ADD_SYMBOL_CLASS.value, 'Add symbol class'),
            (DatabasePermissionFlag.DELETE_SYMBOL_CLASS.value, 'Delete symbol class'),
            (DatabasePermissionFlag.EDIT_SYMBOL_CLASS.value, 'Edit symbol class'),
            (DatabasePermissionFlag.CHANGE_DEFAULT_MODEL_FOR_BOOK_STYLE.value, 'Change default model for book style'),
            (DatabasePermissionFlag.TASKS_LIST.value, 'List tasks'),
            (DatabasePermissionFlag.TASKS_CANCEL.value, 'Cancel a running task'),
            (DatabasePermissionFlag.SET_TRAINING_EPOCHS.value, 'Raise the number of training epochs above the default'),
            (DatabasePermissionFlag.VIEW_SYSTEM_RESOURCES.value, 'View the server resources'),
            (DatabasePermissionFlag.MANAGE_MODELS.value, 'List and delete trained models'),
            (DatabasePermissionFlag.MANAGE_TASK_WORKERS.value, 'Repair the task scheduler and release worker slots'),
        ]

