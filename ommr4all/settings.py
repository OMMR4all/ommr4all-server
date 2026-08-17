import os
import datetime
from typing import NamedTuple, List, Optional

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
import traceback

class TorchImportWatcher:
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch':
            print("\n" + "!" * 60)
            print(f"🚨 TORCH IMPORT DETECTED!")
            print(f"Process ID: {os.getpid()}")
            print("Traceback (How did Torch get here?):")
            for line in traceback.format_stack():
                if "importlib" not in line:
                    print(line.strip())
            print("!" * 60 + "\n")
        return None

#sys.meta_path.insert(0, TorchImportWatcher())
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'x_d3fxz--fi$9ediyggs(re5e&f2^csom_6&+rjw&&%s(doyka'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'channels',
    'daphne',  # makes `manage.py runserver` serve ASGI (websockets); must precede staticfiles
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'restapi.apps.RestApiConfig',
    'webapp.apps.WebappConfig',
    'database.apps.DatabaseConfig',
    'database.lyric_database.apps.LyricsConfig',
    'rest_framework_simplejwt',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ommr4all.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')]
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ommr4all.wsgi.application'
ASGI_APPLICATION = 'ommr4all.routing.application'

# Channel layer for websocket group broadcasts (live chant/document list updates).
# The in-memory layer only works while all traffic is served by a single process
# (manage.py runserver). Behind Apache/mod_wsgi with a separate ASGI server for
# websockets, set REDIS_URL (e.g. redis://localhost:6379/0) so events cross
# process boundaries.
if os.environ.get('REDIS_URL'):
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {'hosts': [os.environ['REDIS_URL']]},
        },
    }
else:
    CHANNEL_LAYERS = {
        'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'},
    }

# Database
# https://docs.djangoproject.com/en/2.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        # overridable via environment (like PRIVATE_MEDIA_ROOT below) so the
        # deployment can place the SQLite file outside the checkout, e.g. on a
        # mounted volume in Docker
        'NAME': os.environ.get('OMMR4ALL_DB_PATH', os.path.join(BASE_DIR, 'db.sqlite3')),
        # web workers and spawned task processes write the book index concurrently;
        # wait out short write locks instead of failing immediately
        'OPTIONS': {
            'timeout': 20,
        },
    }
}

# Page edit locks older than this many hours count as abandoned and are released
# lazily on access (and in bulk by `manage.py release_stale_locks`). 0 disables expiry.
PAGE_EDIT_LOCK_TTL_HOURS = 12


# Password validation
# https://docs.djangoproject.com/en/2.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


LANGUAGES = [
    ('en', 'English'),
    ('de', 'German'),
]

LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale')
]


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.1/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, "static")

PRIVATE_MEDIA_URL = '/storage/'
# overridable via environment so that spawned task worker processes (which
# re-import settings) see the same storage root as the parent, e.g. in tests
PRIVATE_MEDIA_ROOT = os.environ.get('OMMR4ALL_STORAGE_ROOT', os.path.join(BASE_DIR, 'storage'))

DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50 MB


# REST API
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        #'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        # 'rest_framework.authentication.SessionAuthentication',
        # 'rest_framework.authentication.BasicAuthentication',
    ),
}


def jwt_response_payload_handler(token, user=None, request=None):
    return {
        'token': token,
        'permissions': user.get_all_permissions(),
        'is_admin': user.is_superuser or user.is_staff,
    }

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": datetime.timedelta(minutes=120),
    "REFRESH_TOKEN_LIFETIME": datetime.timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": False,
    "BLACKLIST_AFTER_ROTATION": False,
    "UPDATE_LAST_LOGIN": False,

    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "VERIFYING_KEY": "",
    "AUDIENCE": None,
    "ISSUER": None,
    "JSON_ENCODER": None,
    "JWK_URL": None,
    "LEEWAY": 0,

    "AUTH_HEADER_TYPES": ("Bearer",),
    "AUTH_HEADER_NAME": "HTTP_AUTHORIZATION",
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
    "USER_AUTHENTICATION_RULE": "rest_framework_simplejwt.authentication.default_user_authentication_rule",
    "ON_LOGIN_SUCCESS": "rest_framework_simplejwt.serializers.default_on_login_success",
    "ON_LOGIN_FAILED": "rest_framework_simplejwt.serializers.default_on_login_failed",

    "AUTH_TOKEN_CLASSES": ("rest_framework_simplejwt.tokens.AccessToken",),
    "TOKEN_TYPE_CLAIM": "token_type",
    "TOKEN_USER_CLASS": "rest_framework_simplejwt.models.TokenUser",

    "JTI_CLAIM": "jti",

    "SLIDING_TOKEN_REFRESH_EXP_CLAIM": "refresh_exp",
    "SLIDING_TOKEN_LIFETIME": datetime.timedelta(minutes=5),
    "SLIDING_TOKEN_REFRESH_LIFETIME": datetime.timedelta(days=1),

    "TOKEN_OBTAIN_SERIALIZER": "ommr4all.serializer.CustomTokenObtainPairSerializer",
    "TOKEN_REFRESH_SERIALIZER": "rest_framework_simplejwt.serializers.TokenRefreshSerializer",
    "TOKEN_VERIFY_SERIALIZER": "rest_framework_simplejwt.serializers.TokenVerifySerializer",
    "TOKEN_BLACKLIST_SERIALIZER": "rest_framework_simplejwt.serializers.TokenBlacklistSerializer",
    "SLIDING_TOKEN_OBTAIN_SERIALIZER": "rest_framework_simplejwt.serializers.TokenObtainSlidingSerializer",
    "SLIDING_TOKEN_REFRESH_SERIALIZER": "rest_framework_simplejwt.serializers.TokenRefreshSlidingSerializer",
}
JWT_AUTH = {
    'JWT_ENCODE_HANDLER':
    'rest_framework_jwt.utils.jwt_encode_handler',

    'JWT_DECODE_HANDLER':
    'rest_framework_jwt.utils.jwt_decode_handler',

    'JWT_PAYLOAD_HANDLER':
    'rest_framework_jwt.utils.jwt_payload_handler',

    'JWT_PAYLOAD_GET_USER_ID_HANDLER':
    'rest_framework_jwt.utils.jwt_get_user_id_from_payload_handler',

    'JWT_RESPONSE_PAYLOAD_HANDLER':
    jwt_response_payload_handler,

    'JWT_SECRET_KEY': SECRET_KEY,
    'JWT_GET_USER_SECRET_KEY': None,
    'JWT_PUBLIC_KEY': None,
    'JWT_PRIVATE_KEY': None,
    'JWT_ALGORITHM': 'HS256',
    'JWT_VERIFY': True,
    'JWT_VERIFY_EXPIRATION': True,
    'JWT_LEEWAY': 0,
    'JWT_EXPIRATION_DELTA': datetime.timedelta(minutes=120),
    'JWT_AUDIENCE': None,
    'JWT_ISSUER': None,

    'JWT_ALLOW_REFRESH': True,
    'JWT_REFRESH_EXPIRATION_DELTA': datetime.timedelta(days=7),

    'JWT_AUTH_HEADER_PREFIX': 'JWT',
    'JWT_AUTH_COOKIE': None,

}

# LOGGING

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console': {
            'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
        },
    },
    'loggers': {
        'main': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        '__main__': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        'django': {
            'handlers': ['console'],
            'level': 'WARNING',
        },
        'database': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'omr': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'restapi': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}


# RESOURCES

class GPUSettings(NamedTuple):
    # None = detect the cards at startup (see taskresources.resolve_available_gpus);
    # an explicit list (possibly empty) disables the detection
    available_gpus: Optional[List[int]]


def _gpus_from_env() -> Optional[List[int]]:
    """OMMR4ALL_GPUS as a list of device indices.

    Unset or empty means "detect the cards" -- docker compose always defines the
    variable, so a blank value has to keep the default behaviour rather than
    disabling the GPU workers. 'none' turns them off explicitly; otherwise a comma
    separated list such as '0,1'.
    """
    raw = (os.environ.get('OMMR4ALL_GPUS') or '').strip()
    if not raw:
        return None
    if raw.lower() in ('none', 'off'):
        return []
    return [int(i) for i in raw.replace(' ', '').split(',') if i]


GPU_SETTINGS = GPUSettings(_gpus_from_env())


class TaskOperationWatcherSettings(NamedTuple):
    interval: int


# The watcher is the supervisor of the task scheduler (restapi/operationworker/taskwatcher.py):
# it restarts the scheduler and the communicator when they die or stall. It must run, a value
# <= 0 only exists so that tests can construct an unsupervised OperationWorker.
TASK_OPERATION_WATCHER_SETTINGS = TaskOperationWatcherSettings(
    int(os.environ.get('TASK_OPERATION_WATCHER_INTERVAL', '30')),
)

# Task worker processes (restapi/operationworker) are spawned once per
# resource and kept alive between tasks to keep loaded models warm.
# After this many seconds without a task a worker exits to free its
# memory/VRAM; 0 disables the idle shutdown (workers live forever).
TASK_WORKER_IDLE_TIMEOUT = int(os.environ.get('TASK_WORKER_IDLE_TIMEOUT', '600'))

# How long to wait for a worker process to react to each of SIGTERM and SIGKILL before
# giving up on it. A process stuck in uninterruptible sleep (D state, e.g. a wedged GPU
# driver) accepts neither signal, so waiting is capped and the process is then abandoned
# rather than blocking the reaper.
TASK_WORKER_TERMINATE_TIMEOUT = float(os.environ.get('TASK_WORKER_TERMINATE_TIMEOUT', '5'))

# The scheduler loop ticks every 100ms. If its heartbeat is older than this it is
# considered wedged (blocked in a syscall that cannot be interrupted) and restarted.
# Generous compared to the tick rate: a restart is disruptive, a slow loop is not.
TASK_SCHEDULER_STALL_TIMEOUT = float(os.environ.get('TASK_SCHEDULER_STALL_TIMEOUT', '60'))

# Wall-clock limit for a single task, 0 = no limit (the default). Trainings legitimately
# run for hours, so this is opt-in for operators who want a backstop against hung tasks.
TASK_MAX_RUNTIME = float(os.environ.get('TASK_MAX_RUNTIME', '0'))

# Upper bound on the page image handed to a vision language model (text_llm).
# The vision towers of the usual VLMs attend globally over the image patches, so the
# attention matrix grows with the square of the pixel count:
#   bytes ~ n_heads * (pixels / patch^2)^2 * sizeof(dtype)
# A full 2500 px page is enough to ask for tens of GiB. The step normally works on
# color_norm_x2 (20 px per staff line distance, ~2 MP), which fits next to a ~8 GiB
# model on an 11 GiB card; this budget only catches pages whose normalisation blew up.
LLM_MAX_IMAGE_PIXELS = int((os.environ.get('OMMR4ALL_LLM_MAX_PIXELS') or '').strip() or 2_000_000)
