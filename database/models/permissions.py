from django.db import models
from enum import Enum


class DatabasePermissionFlag(Enum):
    ADD_BOOK_STYLE = 'add_book_style'
    DELETE_BOOK_STYLE = 'delete_book_style'
    EDIT_BOOK_STYLE = 'edit_book_style'

    CHANGE_DEFAULT_MODEL_FOR_BOOK_STYLE = 'change_default_model_for_book_style'

    TASKS_LIST = 'tasks_list'
    TASKS_CANCEL = 'tasks_cancel'


class GlobalPermissions(models.Model):
    class Meta:
        permissions = [
            (DatabasePermissionFlag.ADD_BOOK_STYLE.value, 'Add book style'),
            (DatabasePermissionFlag.DELETE_BOOK_STYLE.value, 'Delete book style'),
            (DatabasePermissionFlag.EDIT_BOOK_STYLE.value, 'Edit book style'),
            (DatabasePermissionFlag.CHANGE_DEFAULT_MODEL_FOR_BOOK_STYLE.value, 'Change default model for book style'),
            (DatabasePermissionFlag.TASKS_LIST, 'List tasks'),
            (DatabasePermissionFlag.TASKS_CANCEL, 'Cancel a running task'),
        ]

