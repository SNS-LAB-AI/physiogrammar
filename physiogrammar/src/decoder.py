import numpy as np

from primitives import primitive_reconstruct
from serializer import dequantize_minmax


def parse_header(line):
    fields = line.split("|")
    if not fields or fields[0] != "PG1":
        raise ValueError("Expected PG1 header")
    meta = {}
    for item in fields[1:]:
        key, value = item.split("=", 1)
        meta[key] = value
    return {
        "n": int(meta["n"]),
        "bits": int(meta["bits"]),
        "lo": float(meta["lo"]),
        "hi": float(meta["hi"]),
    }


def decode(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty PhysioGrammar string")

    meta = parse_header(lines[0])
    chunks = []
    for line in lines[1:]:
        fields = line.split("|")
        kind = fields[0]
        params = tuple(int(v) for v in fields[1:])
        chunks.append(primitive_reconstruct(kind, params))

    if chunks:
        qhat = np.concatenate(chunks)[:meta["n"]]
    else:
        qhat = np.zeros(meta["n"], dtype=float)

    qmax = (1 << meta["bits"]) - 1
    qhat = np.clip(qhat, 0, qmax)
    return dequantize_minmax(qhat, meta["lo"], meta["hi"], qmax)
