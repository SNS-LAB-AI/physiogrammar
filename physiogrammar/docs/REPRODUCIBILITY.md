# Reproducibility notes

## Frozen paper settings

The main experiment uses:

```text
window length: 10 s
quantization: 8 bit
candidate boundary: 0.10 s
minimum segment: 0.10 s
lambda: 1, 3, 10, 30, 100, 300, 1000
segment penalty: 1.0
tokenizer: o200k_base
primitives: C, L, E, O, T
```

WESAD uses all discovered subjects with chest ECG and EDA at 700 Hz. BIDMC uses records 01-53 with PPG and impedance respiration at the native 125 Hz sampling rate. Three deterministic windows are placed at approximately 5%, 50% and 95% of the valid start range. Windows containing non-finite samples are skipped.

## Environment

The artifact targets Python 3.13 with versions pinned in `requirements.txt`. The LLM probe uses Ollama 0.34.4 and `llama3.2:3b`.

Record the final machine details before archival release:

```bash
python --version
pip freeze
uname -a
ollama --version
ollama list
```

## Token accounting

Candidate records are cached by text before tokenization. The dynamic program uses standalone record counts as its additive rate surrogate. The final serialized waveform is always tokenized again as a whole string before a token rate or reduction is reported.

Run:

```bash
python scripts/audit_token_boundaries.py --limit 0
```

to save exact counts, independent-record counts and boundary deltas for the full available dataset.

## Expected outputs

`reproduce_v3.py` writes `window_results.csv`, `summary_results.csv` and per-modality Pareto plots.

`reproduce_ablation.py` writes window-level and summary results for token, character and parameter-count objectives, plus matched NRMSE target comparisons.

`reproduce_llm_probe.py` writes row-level answers, aggregate scores, class distributions and chance/majority baselines.

## Archival release

For double-blind review, remove author names, institutional paths and repository owner metadata. After acceptance, create a versioned release and archive the same commit in a permanent repository such as Zenodo.
