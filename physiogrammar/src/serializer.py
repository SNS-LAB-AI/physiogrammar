import numpy as np

from primitives import primitive_text


def quantize_minmax(x, bits=8):
    x = np.asarray(x, dtype=float)
    qmax = (1 << bits) - 1
    lo = float(np.min(x))
    hi = float(np.max(x))
    if abs(hi - lo) < 1e-15:
        q = np.zeros(len(x), dtype=int)
    else:
        q = np.rint((x - lo) / (hi - lo) * qmax).astype(int)
    return q, lo, hi, qmax


def dequantize_minmax(q, lo, hi, qmax):
    q = np.asarray(q, dtype=float)
    if abs(hi - lo) < 1e-15:
        return np.full(len(q), lo, dtype=float)
    return lo + (q / qmax) * (hi - lo)


def header_text(n, bits, lo, hi):
    return f"PG1|n={n}|bits={bits}|lo={lo:.8g}|hi={hi:.8g}"


def serialize(nodes, n, bits, lo, hi):
    header = header_text(n, bits, lo, hi)
    body = "\n".join(node.get("text", primitive_text(node["kind"], node["params"])) for node in nodes)
    return header if not body else header + "\n" + body


def exact_token_count(text, tokenizer):
    return len(tokenizer.encode(text))


def additive_surrogate_count(nodes, tokenizer):
    return sum(len(tokenizer.encode(node.get("text", primitive_text(node["kind"], node["params"])))) for node in nodes)


def token_boundary_audit(text, nodes, tokenizer):
    exact = exact_token_count(text, tokenizer)
    segment_surrogate = additive_surrogate_count(nodes, tokenizer)
    lines = text.splitlines()
    header = lines[0] if lines else ""
    independent = exact_token_count(header, tokenizer)
    independent += sum(exact_token_count("\n" + line, tokenizer) for line in lines[1:])
    return {
        "exact_tokens": exact,
        "segment_surrogate_tokens": segment_surrogate,
        "independent_serialization_tokens": independent,
        "boundary_delta": exact - independent,
    }
