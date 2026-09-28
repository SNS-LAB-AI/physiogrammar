import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decoder import decode
from serializer import quantize_minmax, serialize


class Counter:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def count(self, text):
        return len(self.encode(text))


def test_serializer_decoder_roundtrip_for_nodes():
    x = np.linspace(-1.0, 1.0, 100)
    q, lo, hi, _ = quantize_minmax(x, 8)
    nodes = [{"kind": "L", "params": (100, int(q[0]), int(q[-1])), "text": f"L|100|{int(q[0])}|{int(q[-1])}"}]
    text = serialize(nodes, len(x), 8, lo, hi)
    recon = decode(text)
    assert len(recon) == len(x)
    assert np.all(np.isfinite(recon))
    assert np.isclose(recon[0], lo, atol=1e-6)
    assert np.isclose(recon[-1], hi, atol=1e-6)
