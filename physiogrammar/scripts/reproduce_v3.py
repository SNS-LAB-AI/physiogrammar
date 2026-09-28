import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import amplitude_text, delta_rle_text, quantized_integer_text, raw_csv_text, sax_text
from decoder import decode
from metrics import correlation, nrmse_range, physiology_error, prd
from segmentation import TokenCounter, encode
from serializer import exact_token_count, token_boundary_audit


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_wesad_subjects(root):
    matches = glob.glob(os.path.join(root, "S*", "S*.pkl"))
    subjects = [Path(p).stem for p in matches]
    return sorted(set(subjects), key=lambda s: int(s[1:]))


def load_wesad(root, subject, fs):
    path = os.path.join(root, subject, f"{subject}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    ecg = np.asarray(data["signal"]["chest"]["ECG"]).reshape(-1).astype(float)
    eda = np.asarray(data["signal"]["chest"]["EDA"]).reshape(-1).astype(float)
    return {"ECG": ecg, "EDA": eda}, float(fs)


def load_bidmc(record_number, pn_dir):
    import wfdb

    record_name = f"bidmc{int(record_number):02d}"
    rec = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    names = [str(x).strip().upper().rstrip(",") for x in rec.sig_name]

    def find(candidates):
        for target in candidates:
            for i, name in enumerate(names):
                if target == name or target in name:
                    return i
        return None

    ppg_idx = find(["PLETH", "PPG"])
    resp_idx = find(["RESP", "RESPIRATION", "IMP"])
    if ppg_idx is None or resp_idx is None:
        raise RuntimeError(f"{record_name}: missing PPG or RESP channel")
    return {
        "PPG": np.asarray(rec.p_signal[:, ppg_idx], dtype=float),
        "RESP": np.asarray(rec.p_signal[:, resp_idx], dtype=float),
    }, float(rec.fs)


def choose_windows(x, fs, seconds, count):
    length = int(round(fs * seconds))
    if len(x) < length:
        return []
    max_start = len(x) - length
    if count <= 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(int(0.05 * max_start), int(0.95 * max_start), count).astype(int)
    out = []
    for start in starts:
        window = np.asarray(x[start:start + length], dtype=float)
        if len(window) == length and np.all(np.isfinite(window)):
            out.append((int(start), window))
    return out


def iter_windows(cfg):
    wesad_root = cfg["wesad"]["root"]
    for subject in discover_wesad_subjects(wesad_root):
        signals, fs = load_wesad(wesad_root, subject, cfg["wesad"]["chest_fs"])
        for modality in ["ECG", "EDA"]:
            windows = choose_windows(signals[modality], fs, cfg["window_seconds"], cfg["wesad"]["windows_per_subject"])
            for index, (start, x) in enumerate(windows):
                yield "WESAD", subject, modality, index, start, x, fs

    for record in cfg["bidmc"]["records"]:
        rid = f"bidmc{int(record):02d}"
        try:
            signals, fs = load_bidmc(record, cfg["bidmc"]["pn_dir"])
        except Exception as exc:
            print(f"Skipping {rid}: {exc}")
            continue
        for modality in ["PPG", "RESP"]:
            windows = choose_windows(signals[modality], fs, cfg["window_seconds"], cfg["bidmc"]["windows_per_record"])
            for index, (start, x) in enumerate(windows):
                yield "BIDMC", rid, modality, index, start, x, fs


def add_row(rows, tokenizer, dataset, subject, modality, index, start, method, param, text, raw, recon, fs, audit=None):
    name, phys_error, phys_raw, phys_recon = physiology_error(modality, raw, recon, fs)
    duration = len(raw) / fs
    row = {
        "dataset": dataset,
        "subject": subject,
        "modality": modality,
        "window": index,
        "start_sample": start,
        "fs": fs,
        "duration_s": duration,
        "method": method,
        "param": str(param),
        "tokens": exact_token_count(text, tokenizer),
        "tokens_per_sec": exact_token_count(text, tokenizer) / duration,
        "chars": len(text),
        "nrmse": nrmse_range(raw, recon),
        "prd_pct": prd(raw, recon),
        "corr": correlation(raw, recon),
        "physio_metric": name,
        "physio_error": phys_error,
        "physio_raw": phys_raw,
        "physio_recon": phys_recon,
    }
    if audit:
        row.update({
            "dp_segment_tokens": audit["segment_surrogate_tokens"],
            "independent_serialization_tokens": audit["independent_serialization_tokens"],
            "boundary_delta": audit["boundary_delta"],
        })
    rows.append(row)


def run(config_path, out_dir):
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter(cfg["tokenizer"]["name"])
    rows = []

    for dataset, subject, modality, index, start, x, fs in iter_windows(cfg):
        add_row(rows, counter.tokenizer, dataset, subject, modality, index, start, "RawCSV_TimeAmp", "-", raw_csv_text(x, fs), x, x, fs)
        add_row(rows, counter.tokenizer, dataset, subject, modality, index, start, "RawCSV_AmpOnly", "-", amplitude_text(x), x, x, fs)

        for bits in cfg["baselines"]["quant_bits"]:
            text, recon = quantized_integer_text(x, bits)
            add_row(rows, counter.tokenizer, dataset, subject, modality, index, start, "QuantizedInt", f"{bits}bit", text, x, recon, fs)
            text, recon = delta_rle_text(x, bits)
            add_row(rows, counter.tokenizer, dataset, subject, modality, index, start, "DeltaRLE", f"{bits}bit", text, x, recon, fs)

        for k in cfg["baselines"]["sax_segments"]:
            text, recon = sax_text(x, k, cfg["baselines"]["sax_alphabet"])
            add_row(rows, counter.tokenizer, dataset, subject, modality, index, start, "SAX", f"k={k}", text, x, recon, fs)

        pg = cfg["physiogrammar"]
        for lam in pg["lambdas"]:
            text, nodes, _ = encode(
                x, fs, counter,
                bits=pg["bits"], boundary_sec=pg["boundary_sec"],
                min_seg_sec=pg["min_segment_sec"], lam=lam,
                segment_penalty=pg["segment_penalty"],
                allowed_primitives=pg["primitives"],
            )
            recon = decode(text)
            audit = token_boundary_audit(text, nodes, counter.tokenizer)
            add_row(rows, counter.tokenizer, dataset, subject, modality, index, start, "PhysioGrammar", f"lambda={lam:g}", text, x, recon, fs, audit)

    df = pd.DataFrame(rows)
    keys = ["dataset", "subject", "modality", "window", "start_sample"]
    ref = df[df.method == "RawCSV_TimeAmp"][keys + ["tokens"]].rename(columns={"tokens": "raw_tokens"})
    df = df.merge(ref, on=keys, how="left")
    df["token_reduction_pct"] = 100 * (1 - df["tokens"] / df["raw_tokens"])
    df.to_csv(out / "window_results.csv", index=False)

    summary = df.groupby(["dataset", "modality", "method", "param"], as_index=False).agg(
        n_windows=("nrmse", "size"), n_subjects=("subject", "nunique"),
        tokens_per_sec_mean=("tokens_per_sec", "mean"), token_reduction_pct_mean=("token_reduction_pct", "mean"),
        nrmse_mean=("nrmse", "mean"), prd_pct_mean=("prd_pct", "mean"), corr_mean=("corr", "mean"),
        physio_error_mean=("physio_error", "mean"), boundary_delta_mean=("boundary_delta", "mean"),
    )
    summary.to_csv(out / "summary_results.csv", index=False)

    for (dataset, modality), group in summary.groupby(["dataset", "modality"]):
        plt.figure(figsize=(6.4, 4.6))
        for method, g in group.groupby("method"):
            g = g.sort_values("tokens_per_sec_mean")
            plt.plot(g.tokens_per_sec_mean, g.nrmse_mean, marker="o", label=method)
        plt.xscale("log")
        plt.xlabel("Tokens per second of signal")
        plt.ylabel("NRMSE")
        plt.title(f"{dataset} {modality}")
        plt.grid(alpha=0.25, which="both")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out / f"{dataset}_{modality}_pareto.pdf")
        plt.close()

    print(f"Saved {len(df)} rows to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "v3"))
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
