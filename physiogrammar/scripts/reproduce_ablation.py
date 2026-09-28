import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decoder import decode
from metrics import correlation, nrmse_range, prd
from reproduce_v3 import iter_windows, load_config
from segmentation import TokenCounter, encode
from serializer import exact_token_count


def run(config_path, out_dir):
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter(cfg["tokenizer"]["name"])
    pg = cfg["physiogrammar"]
    rows = []

    for dataset, subject, modality, index, start, x, fs in iter_windows(cfg):
        for mode in cfg["ablation"]["cost_modes"]:
            for lam in pg["lambdas"]:
                text, _, objective = encode(
                    x, fs, counter,
                    bits=pg["bits"], boundary_sec=pg["boundary_sec"],
                    min_seg_sec=pg["min_segment_sec"], lam=lam,
                    segment_penalty=pg["segment_penalty"], cost_mode=mode,
                    allowed_primitives=pg["primitives"],
                )
                recon = decode(text)
                duration = len(x) / fs
                tokens = exact_token_count(text, counter.tokenizer)
                rows.append({
                    "dataset": dataset, "subject": subject, "modality": modality,
                    "window": index, "start_sample": start, "objective": mode,
                    "lambda": lam, "dp_objective": objective,
                    "tokens": tokens, "tokens_per_sec": tokens / duration,
                    "nrmse": nrmse_range(x, recon), "prd_pct": prd(x, recon),
                    "corr": correlation(x, recon),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out / "objective_ablation_window_results.csv", index=False)

    summary = df.groupby(["dataset", "modality", "objective", "lambda"], as_index=False).agg(
        n_windows=("nrmse", "size"), tokens_per_sec_mean=("tokens_per_sec", "mean"),
        nrmse_mean=("nrmse", "mean"), prd_pct_mean=("prd_pct", "mean"), corr_mean=("corr", "mean"),
    )
    summary.to_csv(out / "objective_ablation_summary.csv", index=False)

    target_rows = []
    for (dataset, modality, objective), g in summary.groupby(["dataset", "modality", "objective"]):
        for target in cfg["ablation"]["error_targets"]:
            feasible = g[g.nrmse_mean <= target]
            if len(feasible):
                row = feasible.sort_values(["tokens_per_sec_mean", "nrmse_mean"]).iloc[0]
                target_rows.append({
                    "dataset": dataset, "modality": modality, "objective": objective,
                    "nrmse_target": target, "lambda": row["lambda"],
                    "tokens_per_sec": row["tokens_per_sec_mean"], "nrmse": row["nrmse_mean"],
                })

    targets = pd.DataFrame(target_rows)
    targets.to_csv(out / "objective_ablation_error_targets.csv", index=False)

    advantages = []
    if len(targets):
        for (dataset, modality, target), g in targets.groupby(["dataset", "modality", "nrmse_target"]):
            token = g[g.objective == "token"]
            if len(token) != 1:
                continue
            token_rate = float(token.iloc[0].tokens_per_sec)
            for other in ["char", "param"]:
                comp = g[g.objective == other]
                if len(comp) != 1:
                    continue
                other_rate = float(comp.iloc[0].tokens_per_sec)
                advantages.append({
                    "dataset": dataset, "modality": modality, "nrmse_target": target,
                    "comparison": f"token_vs_{other}", "token_rate": token_rate,
                    "other_rate": other_rate,
                    "token_saving_pct": 100 * (1 - token_rate / other_rate),
                })
    pd.DataFrame(advantages).to_csv(out / "token_objective_advantage.csv", index=False)
    print(f"Saved objective ablation to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "ablation"))
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
