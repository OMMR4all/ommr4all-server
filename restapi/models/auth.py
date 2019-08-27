from mashumaro import DataClassDictMixin
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from django.contrib.auth.models import User, Group


@dataclass
class RestAPIUser(DataClassDictMixin):
    username: Optional[str] = None     # If None, the user is unknown or invalid
    firstName: Optional[str] = ''
    lastName: Optional[str] = ''

    @staticmethod
    def from_user(user: 'User'):
        return RestAPIUser(user.username, user.first_name, user.last_name)


@dataclass
class RestAPIGroup(DataClassDictMixin):
    name: str

    @staticmethod
    def from_group(group: 'Group'):
        return RestAPIGroup(group.name)
