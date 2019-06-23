class JsonParseKeyNotFound(Exception):
    def __init__(self, key: str, d: dict):
        self.key = key
        self.d = d


def require_json(d: dict, key: str):
    if not key in d:
        raise JsonParseKeyNotFound(key, d)

    return d[key]


def optional_enum(d: dict, key: str, enum: type, default):
    if key in d:
        return enum(d[key])
    return default
