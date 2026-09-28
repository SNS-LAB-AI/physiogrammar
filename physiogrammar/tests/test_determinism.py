import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation import encode


class Counter:
    def __init__(self):
        self.cache = {}

    def encode(self, text):
        return list(text.encode("utf-8"))

    def count(self, text):
        if text not in self.cache:
            self.cache[text] = len(self.encode(text))
        return self.cache[text]


def test_encoding_is_deterministic():
    t = np.linspace(0, 4 * np.pi, 250)
    x = np.sin(t) + 0.15 * np.cos(3 * t)
    counter = Counter()
    a = encode(x, 25, counter, lam=30, boundary_sec=0.10, min_seg_sec=0.10)
    b = encode(x, 25, counter, lam=30, boundary_sec=0.10, min_seg_sec=0.10)
    assert a[0] == b[0]
    assert [n["params"] for n in a[1]] == [n["params"] for n in b[1]]
    assert a[2] == b[2]
