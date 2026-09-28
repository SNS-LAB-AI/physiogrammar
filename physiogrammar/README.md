# PhysioGrammar

Reproducibility package for the PhysioGrammar experiments.

PhysioGrammar converts a quantized physiological waveform into a sequence of analytic records. Candidate segments are evaluated with the target tokenizer, then a dynamic program selects a rate-distortion path. Reported token rates are always measured by tokenizing the complete serialized string after encoding.

## Setup

Python 3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The paper configuration is in `configs/paper.yaml`. Set `wesad.root` to a local WESAD directory containing `S*/S*.pkl`. BIDMC records are read through WFDB from PhysioNet.

Primary paper tokenizer: `o200k_base`.

The local language-model probe uses Ollama 0.34.4 with `llama3.2:3b`.

## Reproduce

Cross-dataset experiment:

```bash
python scripts/reproduce_v3.py
```

Objective ablation:

```bash
python scripts/reproduce_ablation.py
```

Exploratory local LLM probe:

```bash
ollama pull llama3.2:3b
python scripts/reproduce_llm_probe.py
```

Tokenizer-boundary audit:

```bash
python scripts/audit_token_boundaries.py --limit 40
```

Use `--limit 0` to audit every available window.

## Tests

```bash
pytest -q
```

The token-accounting test checks the complete serialized string directly. It does not assume that subword token counts are additive across record boundaries.

## Data

This repository does not redistribute WESAD or BIDMC. Obtain the datasets from their original sources and keep their citations and licenses with any derived release.

## Files

- `src/`: encoder, decoder, baselines, metrics and dynamic programming
- `scripts/`: paper experiment entry points
- `tests/`: round-trip, determinism and token-accounting checks
- `configs/paper.yaml`: frozen paper settings
- `docs/`: format, prompt and reproduction notes
