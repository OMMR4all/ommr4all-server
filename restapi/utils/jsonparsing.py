class JsonParseKeyNotFound(Exception):
    def __init__(self, key: str, d: dict):
        self.key = key
        self.d = d


def require_json(d: dict, key: str):
    if not key in d:
        raise JsonParseKeyNotFound(key, d)

    return d[key]

