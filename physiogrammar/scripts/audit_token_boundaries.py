import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_v3 import iter_windows, load_config
from segmentation import TokenCounter, encode
from serializer import token_boundary_audit


def run(config_path, out_path, limit):
    cfg = load_config(config_path)
    counter = TokenCounter(cfg["tokenizer"]["name"])
    pg = cfg["physiogrammar"]
    rows = []

    for widx, (dataset, subject, modality, index, start, x, fs) in enumerate(iter_windows(cfg)):
        if limit and widx >= limit:
            break
        for lam in pg["lambdas"]:
            text, nodes, _ = encode(
                x, fs, counter, bits=pg["bits"], boundary_sec=pg["boundary_sec"],
                min_seg_sec=pg["min_segment_sec"], lam=lam,
                segment_penalty=pg["segment_penalty"], allowed_primitives=pg["primitives"],
            )
            audit = token_boundary_audit(text, nodes, counter.tokenizer)
            rows.append({
                "dataset": dataset, "subject": subject, "modality": modality,
                "window": index, "start_sample": start, "lambda": lam,
                "records": len(nodes), **audit,
                "exact_minus_dp_segment_tokens": audit["exact_tokens"] - audit["segment_surrogate_tokens"],
            })

    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    if len(df):
        print(df[["boundary_delta", "exact_minus_dp_segment_tokens"]].describe().to_string())
        print(f"Nonzero BPE boundary deltas: {(df.boundary_delta != 0).sum()} / {len(df)}")
    print(f"Saved audit to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "token_boundary_audit.csv"))
    parser.add_argument("--limit", type=int, default=40, help="0 audits every available window")
    args = parser.parse_args()
    run(args.config, args.out, args.limit)


if __name__ == "__main__":
    main()
