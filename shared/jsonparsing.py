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


def drop_all_attributes(d: dict, attr: any, recurse=True):
    if attr in d:
        del d[attr]

    if recurse:
        for v in d.values():
            if isinstance(v, list) or isinstance(v, tuple):
                for sub_d in v:
                    drop_all_attributes(sub_d, attr, True)
            elif isinstance(v, dict):
                drop_all_attributes(v, attr, True)
