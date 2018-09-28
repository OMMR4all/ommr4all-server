import numpy as np


def json_to_line(json):
    out = []
    for p in json:
        out.append([p['x'], p['y']])

    return np.array(out, np.int32)


def line_to_json(l: np.ndarray):
    assert(l.ndim == 2)
    assert(l.shape[1] == 2)
    return [{'x': int(l[i, 0]), 'y': int(l[i, 1])} for i in range(l.shape[0])]
