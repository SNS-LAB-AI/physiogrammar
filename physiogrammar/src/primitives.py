import numpy as np
from scipy import signal


def primitive_text(kind, params):
    if kind == "C":
        n, c = params
        return f"C|{n}|{c}"
    if kind == "L":
        n, a, b = params
        return f"L|{n}|{a}|{b}"
    if kind == "E":
        n, c, a, tau = params
        return f"E|{n}|{c}|{a}|{tau}"
    if kind == "O":
        n, c, a, b, k = params
        return f"O|{n}|{c}|{a}|{b}|{k}"
    if kind == "T":
        n, c, a, mu, sigma = params
        return f"T|{n}|{c}|{a}|{mu}|{sigma}"
    raise ValueError(f"Unknown primitive: {kind}")


def primitive_reconstruct(kind, params):
    n = int(params[0])
    t = np.arange(n, dtype=float)

    if kind == "C":
        _, c = params
        return np.full(n, c, dtype=float)

    if kind == "L":
        _, a, b = params
        return np.linspace(a, b, n)

    if kind == "E":
        _, c, a, tau = params
        tau = max(float(tau), 1.0)
        return c + a * np.exp(-t / tau)

    if kind == "O":
        _, c, a, b, k = params
        w = 2 * np.pi * float(k) / max(n, 1)
        return c + a * np.sin(w * t) + b * np.cos(w * t)

    if kind == "T":
        _, c, a, mu, sigma = params
        sigma = max(float(sigma), 1.0)
        return c + a * np.exp(-0.5 * ((t - float(mu)) / sigma) ** 2)

    raise ValueError(f"Unknown primitive: {kind}")


def fit_primitive_candidates(y, allowed=None):
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(n, dtype=float)
    allowed = set(allowed or ["C", "L", "E", "O", "T"])
    out = []

    if "C" in allowed:
        c = int(np.rint(np.mean(y)))
        p = (n, c)
        out.append(("C", p, primitive_reconstruct("C", p)))

    if "L" in allowed and n >= 2:
        slope, intercept = np.polyfit(t, y, 1)
        a = int(np.rint(intercept))
        b = int(np.rint(intercept + slope * (n - 1)))
        p = (n, a, b)
        out.append(("L", p, primitive_reconstruct("L", p)))

    if "E" in allowed and n >= 8:
        best = None
        tau_grid = sorted(set(max(1, int(v)) for v in [n / 12, n / 8, n / 4, n / 2, n, 2 * n]))
        for tau in tau_grid:
            g = np.exp(-t / tau)
            X = np.column_stack([np.ones(n), g])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            c0, a0 = [int(np.rint(v)) for v in coef]
            p = (n, c0, a0, tau)
            pred = primitive_reconstruct("E", p)
            sse = float(np.sum((y - pred) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p, pred)
        if best is not None:
            out.append(("E", best[1], best[2]))

    if "O" in allowed and n >= 16 and np.std(y) > 1e-9:
        yd = signal.detrend(y, type="linear")
        mag = np.abs(np.fft.rfft(yd))
        hi = min(len(mag), max(2, n // 4 + 1))
        if hi > 2:
            k = 1 + int(np.argmax(mag[1:hi]))
            w = 2 * np.pi * k / n
            X = np.column_stack([np.ones(n), np.sin(w * t), np.cos(w * t)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            c0, a0, b0 = [int(np.rint(v)) for v in coef]
            p = (n, c0, a0, b0, k)
            out.append(("O", p, primitive_reconstruct("O", p)))

    if "T" in allowed and n >= 12 and np.std(y) > 1e-9:
        edge = max(2, n // 10)
        base = float(np.median(np.r_[y[:edge], y[-edge:]]))
        mu = int(np.argmax(np.abs(y - base)))
        best = None
        sigma_grid = sorted(set(max(1, int(v)) for v in [n / 30, n / 20, n / 12, n / 8, n / 5]))
        for sigma in sigma_grid:
            g = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
            X = np.column_stack([np.ones(n), g])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            c0, a0 = [int(np.rint(v)) for v in coef]
            p = (n, c0, a0, mu, sigma)
            pred = primitive_reconstruct("T", p)
            sse = float(np.sum((y - pred) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p, pred)
        if best is not None:
            out.append(("T", best[1], best[2]))

    return out
