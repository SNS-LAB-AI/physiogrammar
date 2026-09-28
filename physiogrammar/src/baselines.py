import numpy as np
from scipy import stats

from serializer import quantize_minmax, dequantize_minmax


def raw_csv_text(x, fs):
    return "time,value\n" + "\n".join(
        f"{i / fs:.6f},{v:.6f}" for i, v in enumerate(x)
    )


def amplitude_text(x):
    return ",".join(f"{v:.6f}" for v in x)


def quantized_integer_text(x, bits=8):
    q, lo, hi, qmax = quantize_minmax(x, bits)
    text = f"Q{bits}|n={len(q)}|lo={lo:.8g}|hi={hi:.8g}\n" + ",".join(map(str, q))
    return text, dequantize_minmax(q, lo, hi, qmax)


def _rle_pairs(values):
    values = np.asarray(values, dtype=int)
    if len(values) == 0:
        return []
    out = []
    current = int(values[0])
    count = 1
    for value in values[1:]:
        value = int(value)
        if value == current:
            count += 1
        else:
            out.append((current, count))
            current, count = value, 1
    out.append((current, count))
    return out


def delta_rle_text(x, bits=8):
    q, lo, hi, qmax = quantize_minmax(x, bits)
    if len(q) == 0:
        return "", np.array([])

    runs = _rle_pairs(np.diff(q))
    body = ",".join(f"{value}*{count}" for value, count in runs)
    text = f"DR{bits}|n={len(q)}|lo={lo:.8g}|hi={hi:.8g}|first={int(q[0])}\n{body}"

    values = [int(q[0])]
    current = values[0]
    for delta, count in runs:
        for _ in range(count):
            current += int(delta)
            values.append(current)
    values = np.asarray(values[:len(q)], dtype=int)
    return text, dequantize_minmax(values, lo, hi, qmax)


def _paa(x, n_segments):
    chunks = np.array_split(np.asarray(x, dtype=float), n_segments)
    means = np.array([np.mean(c) if len(c) else 0.0 for c in chunks], dtype=float)
    return means, [len(c) for c in chunks]


def sax_text(x, n_segments=64, alphabet=8):
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    z = (x - mu) / (sd + 1e-12)
    paa, lengths = _paa(z, n_segments)

    breaks = stats.norm.ppf(np.arange(1, alphabet) / alphabet)
    ids = np.digitize(paa, breaks)
    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if alphabet > len(symbols):
        raise ValueError("SAX alphabet must be 26 or smaller")

    word = "".join(symbols[i] for i in ids)
    centers = stats.norm.ppf((np.arange(alphabet) + 0.5) / alphabet)
    zhat = np.concatenate([np.full(length, centers[i]) for i, length in zip(ids, lengths)])[:len(x)]
    recon = zhat * sd + mu
    text = f"SAX|n={len(x)}|k={n_segments}|a={alphabet}|mu={mu:.8g}|sd={sd:.8g}\n{word}"
    return text, recon
