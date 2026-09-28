import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines import amplitude_text, sax_text
from metrics import eda_peak_count, ppg_peak_count, resp_peak_count
from reproduce_v3 import choose_windows, load_bidmc, load_config, load_wesad
from segmentation import TokenCounter, encode


def robust_range(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5)) + 1e-12


def gt_trend(x):
    t = np.arange(len(x), dtype=float)
    slope, _ = np.polyfit(t, x, 1)
    ratio = slope * max(len(x) - 1, 1) / robust_range(x)
    if ratio > 0.15:
        return "UP"
    if ratio < -0.15:
        return "DOWN"
    return "STABLE"


def gt_half_mean(x):
    m = len(x) // 2
    delta = (np.mean(x[m:]) - np.mean(x[:m])) / robust_range(x)
    if delta > 0.08:
        return "SECOND_HIGHER"
    if delta < -0.08:
        return "FIRST_HIGHER"
    return "EQUAL"


def ecg_event_count(x, fs):
    x = np.asarray(x, dtype=float)
    nyq = fs / 2
    b, a = signal.butter(2, [5 / nyq, min(25 / nyq, 0.99)], btype="bandpass")
    xf = signal.filtfilt(b, a, x)
    z = (xf - np.median(xf)) / (np.std(xf) + 1e-12)
    p1, _ = signal.find_peaks(z, distance=max(1, int(0.30 * fs)), prominence=0.7)
    p2, _ = signal.find_peaks(-z, distance=max(1, int(0.30 * fs)), prominence=0.7)
    return int(max(len(p1), len(p2)))


def modality_event_count(modality, x, fs):
    if modality == "ECG":
        return ecg_event_count(x, fs)
    if modality == "EDA":
        return eda_peak_count(x, fs)
    if modality == "PPG":
        return ppg_peak_count(x, fs)
    if modality == "RESP":
        return resp_peak_count(x, fs)
    return 0


def event_bucket(n):
    if n <= 2:
        return "0_2"
    if n <= 5:
        return "3_5"
    if n <= 10:
        return "6_10"
    return "11_PLUS"


def primary_frequency_hz(modality, x, fs):
    y = signal.detrend(np.asarray(x, dtype=float))
    freqs, psd = signal.periodogram(y, fs=fs)
    bands = {"ECG": (0.5, 3.5), "PPG": (0.5, 3.5), "RESP": (0.05, 1.0), "EDA": (0.01, 1.0)}
    lo, hi = bands[modality]
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask) or np.sum(psd[mask]) <= 0:
        return 0.0
    return float(freqs[mask][np.argmax(psd[mask])])


def frequency_bucket(f):
    if f < 0.5:
        return "SLOW"
    if f <= 2.0:
        return "MID"
    return "FAST"


def ground_truth(modality, x, fs):
    return {
        "trend": gt_trend(x),
        "half_mean": gt_half_mean(x),
        "event_bucket": event_bucket(modality_event_count(modality, x, fs)),
        "frequency_band": frequency_bucket(primary_frequency_hz(modality, x, fs)),
    }


def prompt_text(modality, fs, duration, representation, data):
    grammar = """PhysioGrammar legend:
PG1 gives n, bits, lo and hi. C|n|c is constant. L|n|a|b is linear from a to b.
E|n|c|a|tau is c+a*exp(-t/tau). O|n|c|a|b|k is a sinusoidal segment.
T|n|c|a|mu|sig is a Gaussian transient. Segments are consecutive."""
    sax = """SAX legend:
The header gives n, k, alphabet size, mean and standard deviation. A-H are increasing Gaussian SAX bins."""
    if representation == "PhysioGrammar":
        help_text = grammar
    elif representation == "SAX":
        help_text = sax
    else:
        help_text = "RawAmp is the ordered amplitude sequence. Sampling interval is 1/fs."

    return f"""You are analyzing a {duration:.1f}-second {modality} physiological waveform sampled at {fs:g} Hz.

Representation type: {representation}

{help_text}

Answer only one JSON object with exactly these keys and allowed values:
{{
  "trend": "UP" | "DOWN" | "STABLE",
  "half_mean": "FIRST_HIGHER" | "SECOND_HIGHER" | "EQUAL",
  "event_bucket": "0_2" | "3_5" | "6_10" | "11_PLUS",
  "frequency_band": "SLOW" | "MID" | "FAST"
}}

Definitions:
- trend: overall direction across the complete window.
- half_mean: which half has the higher average amplitude; use EQUAL if close.
- event_bucket: estimated number of ECG beats, PPG pulses, EDA responses, or respiration cycles.
- frequency_band: SLOW < 0.5 Hz, MID = 0.5 to 2.0 Hz, FAST > 2.0 Hz.

Do not diagnose disease. Infer only these signal properties.

DATA:
{data}"""


def ollama_generate(cfg, prompt):
    llm = cfg["llm_probe"]
    payload = {
        "model": llm["model"], "prompt": prompt, "stream": False, "format": "json",
        "keep_alive": "15m",
        "options": {
            "temperature": llm["temperature"], "seed": llm["seed"],
            "num_ctx": llm["num_ctx"], "num_predict": llm["num_predict"],
        },
    }
    request = urllib.request.Request(
        llm["ollama_url"], data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama is not reachable") from exc


def panel(cfg):
    out = []
    for subject in cfg["llm_probe"]["wesad_subjects"]:
        signals, fs = load_wesad(cfg["wesad"]["root"], subject, cfg["wesad"]["chest_fs"])
        for modality in ["ECG", "EDA"]:
            windows = choose_windows(signals[modality], fs, cfg["window_seconds"], 3)
            start, x = windows[len(windows) // 2]
            out.append(("WESAD", subject, modality, start, x, fs))

    for record in cfg["llm_probe"]["bidmc_records"]:
        rid = f"bidmc{int(record):02d}"
        signals, fs = load_bidmc(record, cfg["bidmc"]["pn_dir"])
        for modality in ["PPG", "RESP"]:
            windows = choose_windows(signals[modality], fs, cfg["window_seconds"], 3)
            start, x = windows[len(windows) // 2]
            out.append(("BIDMC", rid, modality, start, x, fs))
    return out


def run(config_path, out_dir):
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter(cfg["tokenizer"]["name"])
    rows = []
    truths = []

    for dataset, subject, modality, start, x, fs in panel(cfg):
        truth = ground_truth(modality, x, fs)
        truths.append(truth)
        pg = cfg["physiogrammar"]
        pg_text, _, _ = encode(
            x, fs, counter, bits=pg["bits"], boundary_sec=pg["boundary_sec"],
            min_seg_sec=pg["min_segment_sec"], lam=cfg["llm_probe"]["pg_lambda"][modality],
            segment_penalty=pg["segment_penalty"], allowed_primitives=pg["primitives"],
        )
        sax, _ = sax_text(x, cfg["llm_probe"]["sax_k"][modality], cfg["baselines"]["sax_alphabet"])
        reps = {"RawAmp": amplitude_text(x), "SAX": sax, "PhysioGrammar": pg_text}

        for name, text in reps.items():
            prompt = prompt_text(modality, fs, len(x) / fs, name, text)
            result = ollama_generate(cfg, prompt)
            raw_response = result.get("response", "")
            try:
                pred = json.loads(raw_response)
            except Exception:
                pred = {}

            row = {
                "dataset": dataset, "subject": subject, "modality": modality,
                "start_sample": start, "representation": name,
                "model": cfg["llm_probe"]["model"],
                "model_prompt_tokens": result.get("prompt_eval_count", np.nan),
                "o200k_prompt_tokens": counter.count(prompt),
                "raw_response": raw_response,
            }
            correct = []
            for key in ["trend", "half_mean", "event_bucket", "frequency_band"]:
                value = str(pred.get(key, "")).strip().upper()
                row[f"gt_{key}"] = truth[key]
                row[f"pred_{key}"] = value
                row[f"correct_{key}"] = int(value == truth[key])
                correct.append(row[f"correct_{key}"])
            row["qa_accuracy"] = float(np.mean(correct))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "ai_qa_results.csv", index=False)
    summary = df.groupby("representation", as_index=False).agg(
        n_windows=("qa_accuracy", "size"), model_prompt_tokens_mean=("model_prompt_tokens", "mean"),
        qa_accuracy_mean=("qa_accuracy", "mean"), trend_accuracy=("correct_trend", "mean"),
        half_mean_accuracy=("correct_half_mean", "mean"), event_bucket_accuracy=("correct_event_bucket", "mean"),
        frequency_band_accuracy=("correct_frequency_band", "mean"),
    )
    summary.to_csv(out / "ai_qa_summary.csv", index=False)

    tasks = ["trend", "half_mean", "event_bucket", "frequency_band"]
    majority = {}
    distribution = []
    for task in tasks:
        labels = [item[task] for item in truths]
        counts = Counter(labels)
        majority[task] = max(counts.values()) / len(labels)
        for label, count in sorted(counts.items()):
            distribution.append({"task": task, "label": label, "count": count})
    pd.DataFrame(distribution).to_csv(out / "class_distribution.csv", index=False)
    pd.DataFrame([{
        "majority_accuracy": float(np.mean(list(majority.values()))),
        "uniform_random_accuracy": float(np.mean([1 / 3, 1 / 3, 1 / 4, 1 / 3])),
    }]).to_csv(out / "sanity_baselines.csv", index=False)
    print(f"Saved exploratory LLM probe to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "llm_probe"))
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
