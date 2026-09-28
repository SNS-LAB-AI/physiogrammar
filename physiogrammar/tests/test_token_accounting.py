import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation import TokenCounter, encode
from serializer import exact_token_count, token_boundary_audit


def test_final_token_count_is_whole_string_count():
    pytest.importorskip("tiktoken")
    counter = TokenCounter("o200k_base")
    t = np.linspace(0, 10, 500)
    x = np.sin(2 * np.pi * 1.1 * t) + 0.05 * t
    text, nodes, _ = encode(x, 50, counter, lam=30)
    audit = token_boundary_audit(text, nodes, counter.tokenizer)
    assert audit["exact_tokens"] == exact_token_count(text, counter.tokenizer)
    assert audit["boundary_delta"] == audit["exact_tokens"] - audit["independent_serialization_tokens"]
    assert audit["segment_surrogate_tokens"] >= 0
