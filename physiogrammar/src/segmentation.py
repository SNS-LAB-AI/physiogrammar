import numpy as np

from primitives import fit_primitive_candidates, primitive_text
from serializer import quantize_minmax, serialize


class TokenCounter:
    def __init__(self, name="o200k_base"):
        import tiktoken
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)
        self.cache = {}

    def count(self, text):
        if text not in self.cache:
            self.cache[text] = len(self.tokenizer.encode(text))
        return self.cache[text]

    def encode(self, text):
        return self.tokenizer.encode(text)


def precompute_candidates(q, fs, qmax, token_counter, boundary_sec=0.10,
                          min_seg_sec=0.10, allowed_primitives=None):
    q = np.asarray(q, dtype=float)
    n = len(q)
    step = max(4, int(round(boundary_sec * fs)))
    min_len = max(4, int(round(min_seg_sec * fs)))

    bounds = list(range(0, n, step))
    if not bounds or bounds[-1] != n:
        bounds.append(n)

    models = {}
    for i in range(len(bounds) - 1):
        for j in range(i + 1, len(bounds)):
            a, b = bounds[i], bounds[j]
            if b - a < min_len:
                continue
            y = q[a:b]
            options = []
            for kind, params, pred in fit_primitive_candidates(y, allowed_primitives):
                text = primitive_text(kind, params)
                dist = float(np.sum((y - pred) ** 2) / (qmax * qmax + 1e-12))
                options.append({
                    "kind": kind,
                    "params": tuple(int(v) for v in params),
                    "text": text,
                    "tokens": token_counter.count(text),
                    "chars": len(text),
                    "nparams": len(params),
                    "dist": dist,
                })
            models[(i, j)] = options
    return bounds, models


def solve(bounds, models, lam, segment_penalty=1.0, cost_mode="token"):
    m = len(bounds)
    dp = np.full(m, np.inf)
    back = [None] * m
    dp[0] = 0.0

    def rate_cost(option):
        if cost_mode == "token":
            return float(option["tokens"])
        if cost_mode == "char":
            return float(option["chars"]) / 4.0
        if cost_mode == "param":
            return float(option["nparams"])
        raise ValueError(f"Unknown cost mode: {cost_mode}")

    for j in range(1, m):
        for i in range(j):
            for option in models.get((i, j), []):
                value = dp[i] + rate_cost(option) + lam * option["dist"] + segment_penalty
                if value < dp[j]:
                    dp[j] = value
                    back[j] = (i, option)

    if back[-1] is None:
        raise RuntimeError("No valid PhysioGrammar path found")

    nodes = []
    j = m - 1
    while j > 0:
        i, option = back[j]
        nodes.append(option)
        j = i
    nodes.reverse()
    return nodes, float(dp[-1])


def encode(x, fs, token_counter, bits=8, boundary_sec=0.10,
           min_seg_sec=0.10, lam=30.0, segment_penalty=1.0,
           cost_mode="token", allowed_primitives=None):
    q, lo, hi, qmax = quantize_minmax(x, bits)
    bounds, models = precompute_candidates(
        q, fs, qmax, token_counter,
        boundary_sec=boundary_sec,
        min_seg_sec=min_seg_sec,
        allowed_primitives=allowed_primitives,
    )
    nodes, objective = solve(
        bounds, models, lam,
        segment_penalty=segment_penalty,
        cost_mode=cost_mode,
    )
    text = serialize(nodes, len(x), bits, lo, hi)
    return text, nodes, objective
