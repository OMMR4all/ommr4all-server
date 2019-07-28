from dataclasses import dataclass
from typing import Dict, NamedTuple, TYPE_CHECKING, Union
from enum import IntEnum
import pickle

if TYPE_CHECKING:
    from database import DatabaseBook
    from django.contrib.auth.models import User, Group

_permissions_file = '.permissions.pkl'


class DatabaseBookPermissionFlag(IntEnum):
    NONE = 0
    READ = 1
    WRITE = 2

    EDIT_PERMISSIONS = 4

    ADD_PAGES = 8
    DELETE_PAGES = 16
    RENAME_PAGES = 32

    DELETE_BOOK = 64
    EDIT_BOOK_META = 128

    READ_WRITE = READ | WRITE
    ALLOWED_OTHER_PERMISSIONS = READ_WRITE


@dataclass
class BookPermissionFlags:
    flags: int = DatabaseBookPermissionFlag.NONE

    def to_json(self):
        return {
            'flags': self.flags
        }

    @staticmethod
    def full_access_flags():
        return BookPermissionFlags(2 ** 32 - 1)

    def grant_all(self):
        self.flags = BookPermissionFlags.full_access_flags().flags

    def __or__(self, other: 'BookPermissionFlags'):
        return BookPermissionFlags(self.flags | other.flags)

    def __and__(self, other: 'BookPermissionFlags'):
        return BookPermissionFlags(self.flags & other.flags)

    def has(self, flag: DatabaseBookPermissionFlag):
        return (self.flags & flag) == flag

    def set(self, flag: DatabaseBookPermissionFlag):
        self.flags |= flag

    def erase(self, flag: DatabaseBookPermissionFlag):
        if self.has(flag):
            self.flags -= flag


class BookPermissionData(NamedTuple):
    users: Dict[str, BookPermissionFlags]
    groups: Dict[str, BookPermissionFlags]
    default: BookPermissionFlags

    def to_json(self):
        return {
            'users': dict([(key, val.to_json()) for key, val in self.users.items()]),
            'groups': dict([(key, val.to_json()) for key, val in self.groups.items()]),
            'default': self.default.to_json(),
        }


class DatabaseBookPermissions:
    def write(self):
        with open(self.book.local_path(_permissions_file), 'wb') as f:
            pickle.dump(self.permissions, f)

    @staticmethod
    def load(book: 'DatabaseBook'):
        try:
            with open(book.local_path(_permissions_file), 'rb') as f:
                permissions = pickle.load(f)
        except FileNotFoundError:
            permissions = BookPermissionData({}, {}, BookPermissionFlags())

        return DatabaseBookPermissions(book, permissions)

    def __init__(self, book: 'DatabaseBook', permissions: BookPermissionData):
        self.book = book
        self.permissions = permissions

    def resolve_user_permissions(self, user: 'User'):
        if user.is_superuser:
            # superuser has full access, always
            return BookPermissionFlags.full_access_flags()

        # ensure other flags never allow more than written in ALLOWED_OTHER_PERMISSIONS
        flags = self.permissions.default & BookPermissionFlags(DatabaseBookPermissionFlag.ALLOWED_OTHER_PERMISSIONS)
        un = user.username
        if un in self.permissions.users:
            flags = flags | self.permissions.users[un]

        for group in user.groups.all():
            if group.name in self.permissions.groups:
                flags = flags | self.permissions.groups[group.name]

        return flags

    def get_or_add_user_permissions(self, user: Union['User', str], default: BookPermissionFlags = None) -> BookPermissionFlags:
        name = user if isinstance(user, str) else user.username
        if name not in self.permissions.users:
            self.permissions.users[name] = BookPermissionFlags()

        if default:
            self.permissions.users[name] = default
            self.write()

        return self.permissions.users[name]

    def delete_user_permissions(self, user: Union['User', str]):
        name = user if isinstance(user, str) else user.username
        if name in self.permissions.users:
            del self.permissions.users[name]
            self.write()

    def get_or_add_group_permissions(self, group: Union['Group', str], default: BookPermissionFlags = None) -> BookPermissionFlags:
        name = group if isinstance(group, str) else group.name
        if name not in self.permissions.groups:
            self.permissions.groups[name] = BookPermissionFlags()

        if default:
            self.permissions.groups[name] = default
            self.write()

        return self.permissions.groups[name]

    def delete_group_permissions(self, group: Union['Group', str]):
        name = group if isinstance(group, str) else group.name
        if name in self.permissions.groups:
            del self.permissions.groups[name]
            self.write()
